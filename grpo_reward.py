"""
GRPO Reward Module for Motion Generation

Reward Components:
1. Text-Motion Matching Score: Cosine similarity between text and motion embeddings
2. Physical Plausibility: Foot skating penalty + motion smoothness
3. Numerical Accuracy: Direction-aware step counting, signed rotation,
   temporal phase segmentation, and ordered constraint matching
"""

import re
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from utils.motion_utils import recover_from_ric, recover_root_rot_pos
from spatiotemporal_reward import (
    constraints_to_subgoals,
    evaluate_compositional,
)


# ---------------------------------------------------------------------------
# Data structures for phase-aware analysis
# ---------------------------------------------------------------------------

class Direction(Enum):
    FORWARD = 'forward'
    BACKWARD = 'backward'
    LEFT = 'left'
    RIGHT = 'right'
    ANY = 'any'  # no direction specified in caption

@dataclass
class ConstraintPhase:
    """A parsed constraint from the caption with direction and temporal order."""
    type: str           # 'steps', 'degrees', 'repetitions'
    value: float        # numeric target
    direction: Direction
    order: int          # temporal position (0-based)
    raw: str            # original matched text

@dataclass
class MotionPhase:
    """A detected phase of motion from trajectory analysis."""
    start_frame: int
    end_frame: int
    direction: Direction
    step_count: int
    displacement: float   # meters (XZ plane)
    rotation_deg: float   # signed degrees (positive = left/CCW)
    purity: float = 1.0   # direction purity in [0,1]: cos(actual, ideal_dir)


# ---------------------------------------------------------------------------
# Numerical constraint parser
# ---------------------------------------------------------------------------

# Maps English words to numbers
_WORD2NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'a couple': 2, 'a few': 3, 'several': 4, 'half': 0.5,
}

_WORD_NUM = r'one|two|three|four|five|six|seven|eight|nine|ten'

_NUM_PATTERNS = [
    # sidesteps (must come before generic steps)
    (r'(\d+)\s+(?:side\s*steps?|sidesteps?)', 'steps'),
    (rf'({_WORD_NUM})\s+(?:side\s*steps?|sidesteps?)', 'steps'),
    # "N steps" / "N step" (digits and words)
    (r'(\d+)\s+steps?', 'steps'),
    (rf'({_WORD_NUM})\s+steps?', 'steps'),
    # "N meters/metres" — treat as steps (adult stride ≈ 0.7-0.8m)
    (r'(\d+)\s+met(?:er|re)s?', 'steps'),
    (rf'({_WORD_NUM})\s+met(?:er|re)s?', 'steps'),
    # "a couple/few steps"
    (r'a\s+couple(?:\s+of)?\s+steps?', 'steps_couple'),
    (r'a\s+few\s+steps?', 'steps_few'),
    # "a step" / "a small step"
    (r'\ba\s+(?:small\s+|large\s+|big\s+)?steps?\b', 'steps_one'),
    # "N times"
    (r'(\d+)\s+times?', 'repetitions'),
    (r'(twice)', 'repetitions'),
    (rf'({_WORD_NUM})\s+times?', 'repetitions'),
    # "N degrees"
    (r'(\d+)\s*degrees?', 'degrees'),
    # "half circle" / "half the circle" (must come before "circle")
    (r'half\s+(?:the\s+|a\s+)?circle', 'degrees_half_circle'),
    # "circle" (but not preceded by "half ")
    (r'(?<!half\s)(?:the\s+|a\s+)?circle', 'degrees_full_circle'),
]

# Temporal clause delimiters
_TEMPORAL_SPLIT = re.compile(
    r'\b(?:then|before|after|finally|next|afterwards|and\s+then)\b|[;.]'
)

# Direction patterns — searched within each clause near a numeric match
_DIRECTION_PATTERNS = [
    (re.compile(r'(?:to\s+(?:the\s+)?)?(?:his|her|their|the)\s+right|\bright\b'), Direction.RIGHT),
    (re.compile(r'(?:to\s+(?:the\s+)?)?(?:his|her|their|the)\s+left|\bleft\b'), Direction.LEFT),
    (re.compile(r'\b(?:forward|forwards|ahead)\b'), Direction.FORWARD),
    (re.compile(r'\b(?:backward|backwards)\b'), Direction.BACKWARD),
    (re.compile(r'\bback\b(?!\s+to)'), Direction.BACKWARD),  # "back" but not "back to"
]


def _extract_direction(text: str, match_start: int = -1, match_end: int = -1) -> Direction:
    """Extract movement direction from text, preferring context near the match.

    If match_start/end are given, look within a local window (30 chars) around
    the match first. Fall back to scanning the full text.
    """
    # Local context window around the numeric match
    if match_start >= 0 and match_end >= 0:
        local_start = max(0, match_start - 30)
        local_end = min(len(text), match_end + 30)
        local = text[local_start:local_end]
        for pat, direction in _DIRECTION_PATTERNS:
            if pat.search(local):
                return direction

    # Fall back to full text
    for pat, direction in _DIRECTION_PATTERNS:
        if pat.search(text):
            return direction
    return Direction.ANY


def parse_numerical_constraints(caption: str) -> List[ConstraintPhase]:
    """Extract numerical constraints with direction and temporal ordering.

    Splits caption into temporal clauses, then extracts numeric patterns
    with associated direction from each clause. No deduplication — allows
    "3 steps right, then 3 steps left" to produce two separate constraints.
    """
    text = caption.lower()

    # Split into temporal clauses
    clause_spans = []
    prev_end = 0
    for m in _TEMPORAL_SPLIT.finditer(text):
        if m.start() > prev_end:
            clause_spans.append((prev_end, m.start()))
        prev_end = m.end()
    if prev_end < len(text):
        clause_spans.append((prev_end, len(text)))

    # If no delimiters found, treat entire caption as one clause
    if not clause_spans:
        clause_spans = [(0, len(text))]

    constraints = []
    covered = set()  # character positions already matched (global)

    for order, (c_start, c_end) in enumerate(clause_spans):
        clause = text[c_start:c_end]

        for pattern, ctype in _NUM_PATTERNS:
            for m in re.finditer(pattern, clause):
                # Convert to global positions for overlap check
                g_start = c_start + m.start()
                g_end = c_start + m.end()
                span_range = set(range(g_start, g_end))
                if span_range & covered:
                    continue
                covered |= span_range

                # Extract direction from local context around this match
                match_dir = _extract_direction(clause, m.start(), m.end())

                # Resolve value
                if ctype == 'steps_one':
                    value = 1.0
                    ctype = 'steps'
                elif ctype == 'steps_couple':
                    value = 2.0
                    ctype = 'steps'
                elif ctype == 'steps_few':
                    value = 3.0
                    ctype = 'steps'
                elif ctype == 'degrees_half_circle':
                    value = 180.0
                    ctype = 'degrees'
                elif ctype == 'degrees_full_circle':
                    value = 360.0
                    ctype = 'degrees'
                else:
                    raw = m.group(1) if m.lastindex else m.group(0)
                    if raw == 'twice':
                        value = 2.0
                    elif raw in _WORD2NUM:
                        value = float(_WORD2NUM[raw])
                    else:
                        try:
                            value = float(raw)
                        except ValueError:
                            continue

                constraints.append(ConstraintPhase(
                    type=ctype,
                    value=value,
                    direction=match_dir,
                    order=order,
                    raw=m.group(0),
                ))

    return constraints


# ---------------------------------------------------------------------------
# Motion feature extraction (operates on denormalized motion)
# ---------------------------------------------------------------------------

def _count_steps_in_range(
    foot_contact: torch.Tensor,
    start: int = 0,
    end: int = -1,
) -> int:
    """Count steps from foot contact pattern within a frame range.

    Args:
        foot_contact: [T, 4] binary foot contact (left_heel, left_toe, right_heel, right_toe)
        start: start frame (inclusive)
        end: end frame (exclusive), -1 means T

    Returns:
        Number of detected steps.
    """
    if end == -1:
        end = foot_contact.shape[0]
    fc = foot_contact[start:end]
    if fc.shape[0] < 2:
        return 0

    left = ((fc[:, 0] + fc[:, 1]) > 0.5).cpu().numpy()
    right = ((fc[:, 2] + fc[:, 3]) > 0.5).cpu().numpy()

    steps = 0
    for foot in [left, right]:
        prev = foot[0]
        for t in range(1, len(foot)):
            if not prev and foot[t]:
                steps += 1
            prev = foot[t]
    return steps


# Keep old interface for backward compatibility
def _count_steps(foot_contact: torch.Tensor) -> int:
    return _count_steps_in_range(foot_contact)


def _measure_rotation_signed(
    root_rot_vel: torch.Tensor,
    start: int = 0,
    end: int = -1,
) -> float:
    """Measure signed rotation in degrees. Positive = left/CCW, negative = right/CW.

    Note: HumanML3D uses half-angle quaternion convention, so actual rotation
    is 2x the cumulative rot_vel.

    Args:
        root_rot_vel: [T] root Y-axis rotation velocity (denormalized)
        start: start frame (inclusive)
        end: end frame (exclusive), -1 means T
    """
    if end == -1:
        end = root_rot_vel.shape[0]
    total_rad = root_rot_vel[start:end].sum().item()
    return total_rad * 2.0 * (180.0 / np.pi)  # 2x for half-angle convention


# Keep old interface for backward compatibility
def _measure_rotation(root_rot_vel: torch.Tensor) -> float:
    return abs(_measure_rotation_signed(root_rot_vel))


def _count_repetitions(root_y: torch.Tensor, threshold: float = 0.03) -> int:
    """Count repetitive vertical events (jumps, squats, etc.)

    Detects peaks in root Y position that rise above a threshold
    relative to a running baseline.

    Args:
        root_y: [T] root Y position (denormalized)
        threshold: minimum height delta to count as event

    Returns:
        Number of detected repetitions.
    """
    y = root_y.cpu().numpy()
    baseline = np.median(y)

    # Find peaks above baseline
    above = y > (baseline + threshold)
    count = 0
    in_peak = False
    for v in above:
        if v and not in_peak:
            count += 1
            in_peak = True
        elif not v:
            in_peak = False

    return count


def _foot_skating_score(
    joint_positions: torch.Tensor,
    foot_contact: torch.Tensor,
    fps: float = 20.0,
) -> float:
    """Compute foot skating penalty.

    When a foot is in ground contact, its velocity should be near zero.
    Returns a score in [0, 1] where 1 = no skating.

    Args:
        joint_positions: [T, J, 3] absolute joint positions
        foot_contact: [T, 4] binary contact labels
        fps: frames per second of the motion data

    Returns:
        Score in [0, 1], higher is better.
    """
    T = joint_positions.shape[0]
    if T < 2:
        return 1.0

    # Joint indices: 10 = left foot, 11 = right foot (t2m skeleton)
    left_foot_pos = joint_positions[:, 10, [0, 2]]   # [T, 2] XZ
    right_foot_pos = joint_positions[:, 11, [0, 2]]   # [T, 2] XZ

    # Velocities (m/frame)
    left_vel = torch.norm(left_foot_pos[1:] - left_foot_pos[:-1], dim=-1)   # [T-1]
    right_vel = torch.norm(right_foot_pos[1:] - right_foot_pos[:-1], dim=-1)  # [T-1]

    # Contact masks (use t-1 to align with velocity)
    left_contact = ((foot_contact[:-1, 0] + foot_contact[:-1, 1]) > 0.5).float()
    right_contact = ((foot_contact[:-1, 2] + foot_contact[:-1, 3]) > 0.5).float()

    # Skating = velocity during contact
    left_skating = (left_vel * left_contact).sum()
    right_skating = (right_vel * right_contact).sum()
    contact_frames = left_contact.sum() + right_contact.sum()

    if contact_frames < 1:
        return 1.0

    avg_skating = (left_skating + right_skating) / contact_frames
    # Convert to score: skating of 0 → score 1, skating of 0.05+ → score ~0
    score = torch.exp(-avg_skating * fps * 5.0).item()
    return float(np.clip(score, 0.0, 1.0))


def _smoothness_score(motion: torch.Tensor) -> float:
    """Compute motion smoothness score based on jerk (derivative of acceleration).

    Lower jerk = smoother motion = higher score.

    Args:
        motion: [T, 263] normalized motion

    Returns:
        Score in [0, 1], higher is better.
    """
    T = motion.shape[0]
    if T < 4:
        return 1.0

    # Use global velocity [256:259] for smoothness measurement
    vel = motion[:, 256:259]  # [T, 3]

    # Acceleration
    acc = vel[1:] - vel[:-1]  # [T-1, 3]

    # Jerk
    jerk = acc[1:] - acc[:-1]  # [T-2, 3]

    # Mean jerk magnitude
    jerk_mag = torch.norm(jerk, dim=-1).mean().item()

    # Convert to score: jerk of 0 → 1, high jerk → 0
    # Empirical scale: typical jerk in normalized space ~0.01-0.1
    score = np.exp(-jerk_mag * 20.0)
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Stillness detection (anti-exploit for GRPO)
# ---------------------------------------------------------------------------


def _stillness_score(joint_pos: torch.Tensor, min_displacement: float = 0.3) -> float:
    """Return 0.0 if motion is still, 1.0 if moving enough.

    Uses root-relative joint velocity to detect frozen motion.  This is
    immune to root drift (where repeated identical tokens cause the root
    to slide at constant velocity, faking movement).

    Args:
        joint_pos: [T, 22, 3] 3D joint positions
        min_displacement: minimum root XZ path length (meters) to count as "moving"

    Returns:
        Score in [0, 1]. 0 = completely still, 1 = moving enough.
    """
    T = joint_pos.shape[0]
    if T < 2:
        return 0.0

    # --- Root-relative joint velocity (immune to root drift) ---
    # Subtract root position so constant-velocity drift cancels out.
    rel_pos = joint_pos[:, 1:] - joint_pos[:, 0:1]  # [T, 21, 3]
    rel_vel = torch.norm(rel_pos[1:] - rel_pos[:-1], dim=-1)  # [T-1, 21]
    mean_rel_speed = rel_vel.mean().item()
    # Real motion: ~0.02 m/frame; frozen/repeated tokens: ~0.00001
    rel_score = float(np.clip(mean_rel_speed / 0.005, 0.0, 1.0))

    # --- Joint velocity variance (catches constant-speed drift) ---
    # Frozen motion has near-zero variance even if mean speed is nonzero.
    joint_vel = torch.norm(joint_pos[1:] - joint_pos[:-1], dim=-1)  # [T-1, 22]
    vel_std = joint_vel.std().item()
    # Real motion: std ~0.03; frozen: std ~0.00001
    var_score = float(np.clip(vel_std / 0.005, 0.0, 1.0))

    # --- Root displacement (for locomotion) ---
    root_xz = joint_pos[:, 0, [0, 2]]
    total_path = torch.norm(root_xz[1:] - root_xz[:-1], dim=-1).sum().item()
    root_score = float(np.clip(total_path / min_displacement, 0.0, 1.0))

    # Need EITHER meaningful root displacement OR meaningful relative joint motion.
    # But relative motion must pass — root displacement alone is not enough
    # (it can be faked by constant root velocity from repeated tokens).
    body_score = max(rel_score, var_score)
    return max(body_score * 0.7 + root_score * 0.3, body_score)


# ---------------------------------------------------------------------------
# Phase-aware motion analysis
# ---------------------------------------------------------------------------

def _normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _classify_direction(move_angle_rad: float, initial_facing_rad: float) -> Direction:
    """Classify movement direction relative to character's initial facing.

    In HumanML3D after recover_root_rot_pos: +Z is forward, +X is right.
    atan2(dz, dx) gives the movement angle in the XZ plane.
    """
    relative = _normalize_angle(move_angle_rad - initial_facing_rad)
    deg = np.degrees(relative)
    if -45 <= deg <= 45:
        return Direction.FORWARD
    elif 45 < deg <= 135:
        return Direction.LEFT
    elif -135 <= deg < -45:
        return Direction.RIGHT
    else:
        return Direction.BACKWARD


# Ideal direction angles (relative to initial facing), in radians
# In the (dx, dz) convention where atan2(dz, dx):
#   FORWARD  = facing        (relative angle 0)
#   LEFT     = facing + 90°
#   RIGHT    = facing - 90°
#   BACKWARD = facing + 180°
_IDEAL_RELATIVE_ANGLE = {
    Direction.FORWARD: 0.0,
    Direction.LEFT: np.pi / 2,
    Direction.RIGHT: -np.pi / 2,
    Direction.BACKWARD: np.pi,
}


def _direction_purity(move_angle_rad: float, direction: Direction,
                      initial_facing_rad: float) -> float:
    """Cosine similarity between actual movement and ideal direction axis.

    Returns value in [0, 1]:
      - 1.0 = perfectly aligned
      - 0.71 = 45° off
      - 0.5 = 60° off
      - 0.0 = 90° or more off (or Direction.ANY)
    """
    if direction == Direction.ANY or direction not in _IDEAL_RELATIVE_ANGLE:
        return 1.0
    ideal = initial_facing_rad + _IDEAL_RELATIVE_ANGLE[direction]
    cos_sim = np.cos(_normalize_angle(move_angle_rad - ideal))
    return float(max(0.0, cos_sim))


def _purity_factor(purity: float, threshold: float = 0.6,
                   floor: float = 0.7) -> float:
    """Gentle scoring factor based on direction purity.

    Purity >= threshold (0.6, ~53° off): no penalty, factor = 1.0
    Purity = 0.0 (90°+ off or opposite): factor = floor (0.7)
    Linear interpolation in between.
    """
    if purity >= threshold:
        return 1.0
    return floor + (1.0 - floor) * (purity / threshold)


def analyze_motion_phases(
    motion_raw: torch.Tensor,
    foot_contact: torch.Tensor,
    min_phase_frames: int = 8,
    direction_change_threshold: float = 0.6,
) -> List[MotionPhase]:
    """Segment motion into directional phases.

    Args:
        motion_raw: [T, 263] denormalized motion
        foot_contact: [T, 4] binary foot contact
        min_phase_frames: minimum frames per phase (0.4s at 20fps)
        direction_change_threshold: radians (~35 degrees) to trigger new phase

    Returns:
        List of MotionPhase objects in temporal order.
    """
    T = motion_raw.shape[0]
    if T < min_phase_frames:
        return [MotionPhase(
            start_frame=0, end_frame=T, direction=Direction.ANY,
            step_count=_count_steps_in_range(foot_contact, 0, T),
            displacement=0.0, rotation_deg=0.0,
        )]

    # Recover global root trajectory
    r_rot_quat, r_pos = recover_root_rot_pos(motion_raw.unsqueeze(0))
    root_traj = r_pos.squeeze(0)  # [T, 3]
    global_xz = root_traj[:, [0, 2]]  # [T, 2] — X (right), Z (forward)

    # Initial facing angle (from first frame's rotation quaternion)
    # r_rot_quat: [1, T, 4], quaternion encodes Y-axis rotation
    # At frame 0, rotation angle is 0 → facing +Z → atan2(1, 0) = pi/2
    initial_facing = np.pi / 2  # +Z direction in atan2(z, x) convention

    # Smoothed movement direction
    window = min(5, T // 4)
    if window < 2:
        window = 2
    disp = global_xz[window:] - global_xz[:-window]  # [T-window, 2]
    speed = torch.norm(disp, dim=-1)  # [T-window]
    speed_threshold = 0.005 * window

    # Movement angles: atan2(dz, dx) to match XZ convention
    angles = torch.atan2(disp[:, 1], disp[:, 0])  # [T-window]
    moving = speed > speed_threshold

    # Detect phase boundaries using cumulative angle change over a short window
    boundaries = []
    last_boundary = 0
    lookback = max(3, min_phase_frames // 2)  # compare direction over a few frames
    for t in range(lookback, len(angles)):
        if t - last_boundary < min_phase_frames:
            continue
        if not moving[t].item():
            # Stationary → potential boundary
            boundaries.append(t + window // 2)
            last_boundary = t
            continue
        # Find the last moving frame at least `lookback` frames ago
        ref = max(last_boundary, t - lookback)
        if not moving[ref].item():
            continue
        angle_diff = abs(_normalize_angle(
            (angles[t] - angles[ref]).item()
        ))
        if angle_diff > direction_change_threshold:
            # Place boundary at the midpoint of the transition
            boundaries.append((ref + t) // 2 + window // 2)
            last_boundary = t

    # Build phase list from boundaries
    boundary_frames = [0] + [min(b, T) for b in boundaries] + [T]
    # Remove duplicates and sort
    boundary_frames = sorted(set(boundary_frames))

    phases = []
    root_rot_vel = motion_raw[:, 0]

    for i in range(len(boundary_frames) - 1):
        sf = boundary_frames[i]
        ef = boundary_frames[i + 1]
        if ef - sf < 2:
            continue

        # Phase displacement
        phase_disp = global_xz[min(ef - 1, T - 1)] - global_xz[sf]
        disp_mag = torch.norm(phase_disp).item()

        # Phase direction + purity
        if disp_mag > 0.05:
            move_angle = np.arctan2(phase_disp[1].item(), phase_disp[0].item())
            phase_dir = _classify_direction(move_angle, initial_facing)
            purity = _direction_purity(move_angle, phase_dir, initial_facing)
        else:
            phase_dir = Direction.ANY
            purity = 1.0

        phases.append(MotionPhase(
            start_frame=sf,
            end_frame=ef,
            direction=phase_dir,
            step_count=_count_steps_in_range(foot_contact, sf, ef),
            displacement=disp_mag,
            rotation_deg=_measure_rotation_signed(root_rot_vel, sf, ef),
            purity=purity,
        ))

    # Merge tiny phases into neighbors
    merged = []
    for p in phases:
        if merged and (p.end_frame - p.start_frame) < min_phase_frames:
            # Merge into previous phase — weight purity by displacement
            prev = merged[-1]
            total_disp = prev.displacement + p.displacement
            if total_disp > 1e-6:
                merged_purity = (prev.purity * prev.displacement
                                 + p.purity * p.displacement) / total_disp
            else:
                merged_purity = prev.purity
            merged[-1] = MotionPhase(
                start_frame=prev.start_frame,
                end_frame=p.end_frame,
                direction=prev.direction,  # keep dominant phase's direction
                step_count=prev.step_count + p.step_count,
                displacement=total_disp,
                rotation_deg=prev.rotation_deg + p.rotation_deg,
                purity=merged_purity,
            )
        else:
            merged.append(p)

    return merged if merged else [MotionPhase(
        start_frame=0, end_frame=T, direction=Direction.ANY,
        step_count=_count_steps_in_range(foot_contact, 0, T),
        displacement=0.0, rotation_deg=0.0,
    )]


# ---------------------------------------------------------------------------
# Direction sequence matching (no numeric constraints needed)
# ---------------------------------------------------------------------------

# Direction patterns for direction-sequence matching (stricter than _DIRECTION_PATTERNS).
# Require a locomotion verb nearby to avoid matching body-part references
# like "right hand" or "left arm".
_MOTION_VERBS = r'(?:walk|walks|walking|run|runs|running|jog|jogs|jogging|step|steps|stepping|move|moves|moving|go|goes|going|turn|turns|turning|stumble|stumbles|stumbling|shuffle|shuffles|shuffling|slide|slides|sliding|sidestep|sidesteps|sidestepping|march|marches|marching|stride|strides|striding|lunge|lunges|lunging|hop|hops|hopping|jump|jumps|jumping|skip|skips|skipping|crawl|crawls|crawling)'

_DIR_SEQ_PATTERNS = [
    # "walks/steps/moves forward", "walks to the right", etc.
    # Allow a few words between verb and direction (e.g. "walks sideways to the left")
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:to\s+(?:the\s+)?)?(?:his|her|their|the\s+)?right'), Direction.RIGHT),
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:to\s+(?:the\s+)?)?(?:his|her|their|the\s+)?left'), Direction.LEFT),
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:forward|forwards|ahead)'), Direction.FORWARD),
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:backward|backwards)'), Direction.BACKWARD),
    # "verb + back" but NOT "back to" (which means "return to")
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}back(?!\s+to)\b'), Direction.BACKWARD),
    # standalone direction at clause start (after temporal split)
    (re.compile(r'(?:^|,\s*)\s*(?:to\s+(?:the\s+)?)?(?:forward|forwards|ahead)\b'), Direction.FORWARD),
    (re.compile(r'(?:^|,\s*)\s*(?:to\s+(?:the\s+)?)?(?:backward|backwards)\b'), Direction.BACKWARD),
    (re.compile(r'(?:^|,\s*)\s*(?:to\s+(?:the\s+)?)(?:right)\b'), Direction.RIGHT),
    (re.compile(r'(?:^|,\s*)\s*(?:to\s+(?:the\s+)?)(?:left)\b'), Direction.LEFT),
]


def _extract_direction_strict(clause: str) -> Direction:
    """Extract movement direction from a clause using strict verb+direction patterns.

    Unlike _extract_direction, this avoids matching body-part references
    like 'right hand' or 'left arm'.
    """
    for pat, direction in _DIR_SEQ_PATTERNS:
        if pat.search(clause):
            return direction
    return Direction.ANY


def parse_direction_sequence(caption: str) -> List[Direction]:
    """Extract ordered direction sequence from caption.

    Works on captions WITHOUT numeric constraints, e.g.:
      "a person walks forward then walks backward" -> [FORWARD, BACKWARD]
      "a person jogs to the left and then turns right" -> [LEFT, RIGHT]

    Uses strict verb+direction patterns to avoid false positives from
    body-part references ("raises right hand" should NOT match).

    Returns empty list if no directions found.
    """
    text = caption.lower()

    # Split into temporal clauses
    clause_spans = []
    prev_end = 0
    for m in _TEMPORAL_SPLIT.finditer(text):
        if m.start() > prev_end:
            clause_spans.append((prev_end, m.start()))
        prev_end = m.end()
    if prev_end < len(text):
        clause_spans.append((prev_end, len(text)))
    if not clause_spans:
        clause_spans = [(0, len(text))]

    directions = []
    for c_start, c_end in clause_spans:
        clause = text[c_start:c_end]
        d = _extract_direction_strict(clause)
        if d != Direction.ANY:
            directions.append(d)

    return directions


def score_direction_sequence(
    directions: List[Direction],
    phases: List[MotionPhase],
) -> float:
    """Score how well motion phases match the expected direction sequence.

    Uses greedy sequential matching: for each expected direction, find the
    next phase (in order) that matches. Score = fraction matched, with:
      - a bonus for correct ordering of the first direction
      - a penalty for redundant extra phases (prevents fidgeting/extra moves)

    Returns score in [0, 1].
    """
    if not directions:
        return 0.0  # no directions to match

    sig_phases = [p for p in phases if p.direction != Direction.ANY and p.displacement > 0.05]
    if not sig_phases:
        return 0.0  # no significant phases — let physical_scores handle stillness

    matched = 0
    last_ph = -1
    purity_sum = 0.0
    for d in directions:
        for ph_i in range(last_ph + 1, len(sig_phases)):
            if sig_phases[ph_i].direction == d:
                matched += 1
                purity_sum += _purity_factor(sig_phases[ph_i].purity)
                last_ph = ph_i
                break

    # Base: fraction matched, each match weighted by its purity factor
    base_score = purity_sum / len(directions)

    # Bonus: if first expected direction matches first significant phase,
    # scaled by that phase's purity (so diagonal first move gets less bonus)
    if sig_phases[0].direction == directions[0]:
        base_score = min(1.0,
                         base_score + 0.15 * _purity_factor(sig_phases[0].purity))

    # Redundancy penalty: extra phases beyond what was requested are
    # penalized to discourage fidgeting / extra body movements.
    # Each extra phase costs 0.1, capped at 0.4 total.
    n_extra = max(0, len(sig_phases) - len(directions))
    redundancy_penalty = min(0.4, 0.1 * n_extra)
    base_score = max(0.0, base_score - redundancy_penalty)

    return base_score


# ---------------------------------------------------------------------------
# Phase-aware constraint scoring
# ---------------------------------------------------------------------------

def _step_accuracy(generated: float, target: float) -> float:
    """Asymmetric step-count accuracy.

    Undershoot is penalized harder than overshoot — the model was observed to
    hesitate and produce fewer steps than requested. Overshoot of 1 is
    tolerated; undershoot of 1 costs ~0.4 in score.

    sigma_under = max(target*0.15, 0.5) → tight penalty for under
    sigma_over  = max(target*0.35, 1.5) → loose tolerance for over
    """
    diff = generated - target
    if diff < 0:
        sigma = max(target * 0.15, 0.5)
    else:
        sigma = max(target * 0.35, 1.5)
    return float(np.exp(-0.5 * (diff / sigma) ** 2))


def score_constraints_against_phases(
    constraints: List[ConstraintPhase],
    phases: List[MotionPhase],
    total_steps: int,
    total_rotation_deg: float,
    total_repetitions: int,
) -> float:
    """Score parsed constraints against detected motion phases.

    Handles two modes:
    - No temporal ordering: score against global totals with direction bonus
    - With temporal ordering: align constraint groups to phase groups, score per-group
    """
    if not constraints:
        return 0.0

    # Check if there's temporal ordering
    orders = set(c.order for c in constraints)
    has_temporal = len(orders) > 1

    if not has_temporal:
        return _score_global(constraints, phases, total_steps,
                             total_rotation_deg, total_repetitions)
    else:
        return _score_temporal(constraints, phases, total_steps,
                               total_rotation_deg, total_repetitions)


def _score_global(
    constraints: List[ConstraintPhase],
    phases: List[MotionPhase],
    total_steps: int,
    total_rotation_deg: float,
    total_repetitions: int,
) -> float:
    """Score constraints without temporal ordering."""
    scores = []
    for c in constraints:
        if c.type == 'steps':
            if c.direction != Direction.ANY and phases:
                # Sum steps from phases matching this direction
                dir_steps = sum(p.step_count for p in phases
                                if p.direction == c.direction)
                # Also consider total if no phases match direction
                generated = dir_steps if dir_steps > 0 else total_steps
            else:
                generated = total_steps
            acc = _step_accuracy(generated, c.value)
            # Direction match is a multiplicative gate, not an additive bonus.
            # Wrong direction caps accuracy at 0.7; correct direction preserves it.
            if c.direction != Direction.ANY:
                matching = [p for p in phases if p.direction == c.direction]
                if matching:
                    # Weight purity by phase displacement
                    total_disp = sum(p.displacement for p in matching)
                    if total_disp > 1e-6:
                        avg_purity = sum(p.purity * p.displacement
                                         for p in matching) / total_disp
                    else:
                        avg_purity = 1.0
                    acc = acc * _purity_factor(avg_purity)
                else:
                    acc = acc * 0.7  # no phase matches direction
            scores.append(acc)

        elif c.type == 'degrees':
            if c.direction == Direction.LEFT:
                generated = total_rotation_deg  # positive = left
            elif c.direction == Direction.RIGHT:
                generated = -total_rotation_deg  # flip sign for right
            else:
                generated = abs(total_rotation_deg)
            sigma = max(c.value * 0.2, 15.0)
            acc = np.exp(-0.5 * ((generated - c.value) / sigma) ** 2)
            scores.append(acc)

        elif c.type == 'repetitions':
            sigma = max(c.value * 0.3, 1.0)
            acc = np.exp(-0.5 * ((total_repetitions - c.value) / sigma) ** 2)
            scores.append(acc)

    return min(1.0, float(np.mean(scores))) if scores else 0.0


def _score_temporal(
    constraints: List[ConstraintPhase],
    phases: List[MotionPhase],
    total_steps: int,
    total_rotation_deg: float,
    total_repetitions: int,
) -> float:
    """Score constraints with temporal ordering against phase sequence."""
    # Group constraints by temporal order
    from collections import defaultdict
    order_groups = defaultdict(list)
    for c in constraints:
        order_groups[c.order].append(c)
    sorted_orders = sorted(order_groups.keys())
    constraint_groups = [order_groups[o] for o in sorted_orders]
    n_groups = len(constraint_groups)

    if not phases or n_groups == 0:
        return _score_global(constraints, phases, total_steps,
                             total_rotation_deg, total_repetitions)

    # Align phases to constraint groups proportionally by frame count
    total_frames = sum(p.end_frame - p.start_frame for p in phases)
    frames_per_group = total_frames / n_groups if n_groups > 0 else total_frames

    phase_groups: List[List[MotionPhase]] = [[] for _ in range(n_groups)]
    cumulative = 0
    group_idx = 0
    for p in phases:
        phase_groups[group_idx].append(p)
        cumulative += (p.end_frame - p.start_frame)
        if cumulative >= frames_per_group * (group_idx + 1) and group_idx < n_groups - 1:
            group_idx += 1

    # Score each constraint group against its aligned phase group
    group_scores = []
    temporal_dir_matches = 0
    temporal_dir_total = 0

    for cg, pg in zip(constraint_groups, phase_groups):
        pg_steps = sum(p.step_count for p in pg)
        pg_rotation = sum(p.rotation_deg for p in pg)
        pg_reps = total_repetitions  # repetitions are hard to segment

        # Dominant direction of phase group
        if pg:
            dir_counts: Dict[Direction, float] = {}
            for p in pg:
                d = p.direction
                dir_counts[d] = dir_counts.get(d, 0) + p.displacement
            pg_dir = max(dir_counts, key=dir_counts.get) if dir_counts else Direction.ANY
        else:
            pg_dir = Direction.ANY

        for c in cg:
            if c.type == 'steps':
                if c.direction != Direction.ANY:
                    dir_steps = sum(p.step_count for p in pg
                                    if p.direction == c.direction)
                    generated = dir_steps if dir_steps > 0 else pg_steps
                else:
                    generated = pg_steps
                acc = _step_accuracy(generated, c.value)
                if c.direction != Direction.ANY:
                    temporal_dir_total += 1
                    matching = [p for p in pg if p.direction == c.direction]
                    if matching and pg_dir == c.direction:
                        temporal_dir_matches += 1
                    if matching:
                        total_disp = sum(p.displacement for p in matching)
                        if total_disp > 1e-6:
                            avg_purity = sum(p.purity * p.displacement
                                             for p in matching) / total_disp
                        else:
                            avg_purity = 1.0
                        acc = acc * _purity_factor(avg_purity)
                    else:
                        acc = acc * 0.7  # direction mismatch
                group_scores.append(acc)

            elif c.type == 'degrees':
                if c.direction == Direction.LEFT:
                    generated = pg_rotation
                elif c.direction == Direction.RIGHT:
                    generated = -pg_rotation
                else:
                    generated = abs(pg_rotation)
                sigma = max(c.value * 0.2, 15.0)
                acc = np.exp(-0.5 * ((generated - c.value) / sigma) ** 2)
                group_scores.append(acc)

            elif c.type == 'repetitions':
                sigma = max(c.value * 0.3, 1.0)
                acc = np.exp(-0.5 * ((pg_reps - c.value) / sigma) ** 2)
                group_scores.append(acc)

    base_score = float(np.mean(group_scores)) if group_scores else 0.0

    # Temporal order bonus: fraction of directional constraints that matched
    temporal_bonus = 0.0
    if temporal_dir_total > 0:
        temporal_bonus = 0.1 * (temporal_dir_matches / temporal_dir_total)

    # Redundancy penalty: significant phases beyond expected groups indicate
    # fidgeting / extra body moves. Each extra costs 0.08, capped at 0.3.
    sig_phases = [p for p in phases if p.direction != Direction.ANY and p.displacement > 0.05]
    n_extra = max(0, len(sig_phases) - n_groups)
    redundancy_penalty = min(0.3, 0.08 * n_extra)

    return max(0.0, min(1.0, base_score + temporal_bonus - redundancy_penalty))

class GRPORewardModel:
    """
    Reward model for GRPO training.

    Combines:
    1. Text-motion matching (cosine similarity from pretrained evaluator)
    2. Physical plausibility (foot skating + smoothness)
    3. Numerical accuracy (step count, rotation, repetitions)
    """

    def __init__(
        self,
        eval_wrapper: EvaluatorModelWrapper,
        vqvae_model,
        word_vectorizer: WordVectorizer,
        device: str = 'cuda:0',
        normalize_reward: bool = True,
        reward_scale: float = 1.0,
        length_penalty_weight: float = 0.0,
        tau: float = 0.1,
        # New reward weights
        physical_weight: float = 0.3,
        numerical_weight: float = 0.5,
        # LLM for caption classification (stillness penalty)
        llm=None,
        tokenizer=None,
    ):
        self.eval_wrapper = eval_wrapper
        self.vqvae = vqvae_model
        self.w_vectorizer = word_vectorizer
        self.device = device
        self.normalize_reward = normalize_reward
        self.reward_scale = reward_scale
        self.length_penalty_weight = length_penalty_weight
        self.tau = tau
        self.physical_weight = physical_weight
        self.numerical_weight = numerical_weight

        # LLM-based caption classification
        self.llm = llm
        self.tokenizer = tokenizer
        self._motion_caption_cache: Dict[str, bool] = {}

        self.vqvae.eval()

        # Load denormalization statistics
        meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        self._mean = torch.from_numpy(
            np.load(f'{meta_dir}/mean.npy')
        ).float().to(device)
        self._std = torch.from_numpy(
            np.load(f'{meta_dir}/std.npy')
        ).float().to(device)

        self._reward_stats = {}

    def _denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        """Denormalize motion from VQ-VAE output space to original space."""
        return motion * self._std + self._mean

    def _is_motion_caption(self, caption: str) -> bool:
        """Use Gemma-2 to judge whether caption describes physical movement.

        Results are cached so each unique caption is only classified once.
        Falls back to True (assume motion) if LLM is unavailable.
        """
        if caption in self._motion_caption_cache:
            return self._motion_caption_cache[caption]

        if self.llm is None or self.tokenizer is None:
            # No LLM available — conservatively assume motion
            self._motion_caption_cache[caption] = True
            return True

        prompt = (
            f'Does the following sentence describe a person physically moving '
            f'their body (e.g. walking, running, jumping, kicking)?\n'
            f'Sentence: "{caption}"\n'
            f'Answer only "yes" or "no":'
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        try:
            with torch.no_grad():
                self.llm.disable_adapter_layers()
                out = self.llm.generate(
                    input_ids, max_new_tokens=3, do_sample=False,
                )
        except Exception:
            # LLM failed — assume motion to be safe
            self._motion_caption_cache[caption] = True
            return True
        finally:
            self.llm.enable_adapter_layers()
        answer = self.tokenizer.decode(out[0, len(input_ids[0]):], skip_special_tokens=True)
        is_motion = 'yes' in answer.lower()
        self._motion_caption_cache[caption] = is_motion
        return is_motion

    @torch.no_grad()
    def compute_reward(
        self,
        captions: List[str],
        motion_tokens_list: List[torch.Tensor],
        return_components: bool = False,
    ) -> torch.Tensor:
        batch_size = len(captions)
        assert len(motion_tokens_list) == batch_size

        # --- Decode motion tokens ---
        motions, motion_lengths = self._decode_motion_tokens(motion_tokens_list)

        # --- Text-motion matching reward (existing) ---
        word_embeddings, pos_one_hots, sent_lens = self._encode_text(captions)

        sent_lens_np = sent_lens.cpu().numpy()
        sorted_indices = np.argsort(-sent_lens_np)
        unsort_indices = np.argsort(sorted_indices)

        word_embeddings_s = word_embeddings[sorted_indices]
        pos_one_hots_s = pos_one_hots[sorted_indices]
        sent_lens_s = sent_lens[sorted_indices]
        motions_s = motions[sorted_indices]
        motion_lengths_s = motion_lengths[sorted_indices]

        text_emb, motion_emb = self.eval_wrapper.get_co_embeddings(
            word_embeddings_s, pos_one_hots_s, sent_lens_s,
            motions_s, motion_lengths_s,
        )

        text_emb = text_emb[unsort_indices]
        motion_emb = motion_emb[unsort_indices]

        matching_scores = self._compute_matching_score(text_emb, motion_emb)

        # --- Physical plausibility reward ---
        physical_scores = torch.zeros(batch_size, device=self.device)

        # --- Numerical accuracy reward ---
        numerical_scores = torch.zeros(batch_size, device=self.device)
        has_numerical = torch.zeros(batch_size, device=self.device)

        # --- Direction sequence reward (covers captions without numbers) ---
        direction_scores = torch.zeros(batch_size, device=self.device)
        has_direction = torch.zeros(batch_size, device=self.device)

        # --- Kinematic reward (spatiotemporal) ---
        kinematic_scores = torch.zeros(batch_size, device=self.device)
        has_kinematic = torch.zeros(batch_size, device=self.device)

        for i in range(batch_size):
            length = int(motion_lengths[i].item())
            motion_norm = motions[i, :length]  # [T, 263]
            motion_raw = self._denormalize(motion_norm)

            # -- Physical plausibility (simplified: stillness only) --
            # Foot contact is binary {0,1} in normalized space; denormalization
            # maps it to ~0.85 which breaks thresholding.  Read from normalized.
            foot_contact = (motion_norm[:, 259:263] > 0.5).float()

            # Recover 3D joint positions
            joint_pos = recover_from_ric(
                motion_raw.unsqueeze(0), joints_num=22
            ).squeeze(0)  # [T, 22, 3]

            # Stillness penalty: if caption describes motion but body is still,
            # apply a smooth penalty. Linear mapping: stillness 0→-1, 1→+1.
            # Avoids the discontinuous jump of the old threshold-based approach.
            stillness = _stillness_score(joint_pos)
            if self._is_motion_caption(captions[i]):
                physical_scores[i] = 2.0 * stillness - 1.0  # [-1.0, 1.0]
            else:
                physical_scores[i] = 1.0  # non-motion captions: no penalty

            # -- Numerical accuracy (phase-aware) --
            constraints = parse_numerical_constraints(captions[i])

            # Analyze motion phases (shared by numerical + direction scoring)
            phases = analyze_motion_phases(
                motion_raw, foot_contact,
                min_phase_frames=8,
                direction_change_threshold=0.6,
            )

            # Skip precision rewards for frozen motion — phases are unreliable
            # when the body isn't actually moving (root drift fakes displacement).
            motion_is_alive = stillness >= 0.5

            if constraints and motion_is_alive:
                has_numerical[i] = 1.0

                # Global fallback values
                total_steps = _count_steps_in_range(foot_contact)
                total_rotation = _measure_rotation_signed(motion_raw[:, 0])
                total_reps = _count_repetitions(motion_raw[:, 3])

                # Phase-aware scoring
                numerical_scores[i] = score_constraints_against_phases(
                    constraints, phases,
                    total_steps=total_steps,
                    total_rotation_deg=total_rotation,
                    total_repetitions=total_reps,
                )

            # -- Direction sequence matching --
            # Works even without numeric constraints: "walks forward then backward"
            dir_seq = parse_direction_sequence(captions[i])
            if dir_seq and not constraints and motion_is_alive:
                # Only use direction reward when numerical is absent,
                # to avoid double-counting (numerical already checks direction)
                has_direction[i] = 1.0
                direction_scores[i] = score_direction_sequence(dir_seq, phases)

            # -- Kinematic reward (spatiotemporal) --
            # Convert constraints → SubGoals via smart adapter, then
            # evaluate against 3D joint kinematics directly.
            subgoals = constraints_to_subgoals(captions[i], constraints) if motion_is_alive else []
            if subgoals:
                has_kinematic[i] = 1.0
                try:
                    kinematic_scores[i] = evaluate_compositional(
                        joint_pos, motion_raw, foot_contact, subgoals,
                    )
                except Exception:
                    kinematic_scores[i] = 0.0

        # --- Combine rewards ---
        # Matching score: shifted to [0, 1] range using positive cosine similarity
        # (InfoNCE is in [-log(B), 0]; instead use raw cosine for combination)
        text_norm = F.normalize(text_emb, p=2, dim=-1)
        motion_norm = F.normalize(motion_emb, p=2, dim=-1)
        cos_sim = (text_norm * motion_norm).sum(dim=-1)  # [B] in [-1, 1]
        cos_sim_01 = (cos_sim + 1.0) / 2.0  # shift to [0, 1]

        # Base reward: cosine similarity (dense, all samples)
        # Downweight cosine to prevent it from dominating — it's too coarse
        # to distinguish "2 steps" from "6 steps" in embedding space.
        kinematic_w = 0.3 * self.numerical_weight
        numerical_w = self.numerical_weight - kinematic_w
        cos_weight = 0.5  # reduced from implicit 1.0
        direction_w = 0.4  # direction matching for captions without numbers

        rewards = (
            cos_weight * cos_sim_01
            + self.physical_weight * physical_scores
            + numerical_w * numerical_scores * has_numerical
            + kinematic_w * kinematic_scores * has_kinematic
            + direction_w * direction_scores * has_direction
        )

        # Store stats
        self._reward_stats = {
            'pos_sim_mean': cos_sim.mean().item(),
            'neg_sim_mean': self._reward_stats.get('neg_sim_mean', 0.0),
            'physical_mean': physical_scores.mean().item(),
            'numerical_mean': (
                numerical_scores[has_numerical > 0].mean().item()
                if has_numerical.sum() > 0 else 0.0
            ),
            'numerical_frac': has_numerical.mean().item(),
            'kinematic_mean': (
                kinematic_scores[has_kinematic > 0].mean().item()
                if has_kinematic.sum() > 0 else 0.0
            ),
            'kinematic_frac': has_kinematic.mean().item(),
            'direction_mean': (
                direction_scores[has_direction > 0].mean().item()
                if has_direction.sum() > 0 else 0.0
            ),
            'direction_frac': has_direction.mean().item(),
        }

        if self.normalize_reward:
            rewards = torch.tanh(rewards)
        rewards = rewards * self.reward_scale

        if return_components:
            return rewards, {
                'matching_scores': cos_sim,
                'physical_scores': physical_scores,
                'numerical_scores': numerical_scores,
                'kinematic_scores': kinematic_scores,
                'direction_scores': direction_scores,
                'has_numerical': has_numerical,
                'has_kinematic': has_kinematic,
                'has_direction': has_direction,
                'reward_stats': self._reward_stats,
            }

        return rewards

    def _decode_motion_tokens(
        self,
        motion_tokens_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode motion tokens to continuous sequences using VQ-VAE decoder."""
        batch_size = len(motion_tokens_list)
        decoded_motions = [None] * batch_size
        motion_lengths = [0] * batch_size

        length_groups = {}
        for i, tokens in enumerate(motion_tokens_list):
            t_len = len(tokens)
            if t_len not in length_groups:
                length_groups[t_len] = []
            length_groups[t_len].append(i)

        for t_len, indices in length_groups.items():
            batch_tokens = torch.stack(
                [motion_tokens_list[i].unsqueeze(0) if motion_tokens_list[i].dim() == 1
                 else motion_tokens_list[i] for i in indices]
            ).to(self.device)
            if batch_tokens.dim() == 3:
                batch_tokens = batch_tokens.squeeze(1)

            try:
                batch_motion = self.vqvae.forward_decoder(batch_tokens)
                for j, idx in enumerate(indices):
                    decoded_motions[idx] = batch_motion[j]
                    motion_lengths[idx] = batch_motion.shape[1]
            except Exception:
                for idx in indices:
                    tokens = motion_tokens_list[idx]
                    if tokens.dim() == 1:
                        tokens = tokens.unsqueeze(0)
                    tokens = tokens.to(self.device)
                    try:
                        motion = self.vqvae.forward_decoder(tokens)
                        decoded_motions[idx] = motion.squeeze(0)
                        motion_lengths[idx] = motion.shape[1]
                    except Exception:
                        dummy_motion = torch.zeros(4, 263, device=self.device)
                        decoded_motions[idx] = dummy_motion
                        motion_lengths[idx] = 4

        max_len = max(motion_lengths)
        motion_dim = decoded_motions[0].shape[-1]
        padded_motions = torch.zeros(batch_size, max_len, motion_dim, device=self.device)

        for i, motion in enumerate(decoded_motions):
            cur_len = motion.shape[0]
            padded_motions[i, :cur_len] = motion

        motion_lengths = torch.tensor(motion_lengths, device=self.device)
        return padded_motions, motion_lengths

    def _encode_text(
        self,
        captions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text captions using word vectorizer."""
        batch_size = len(captions)
        word_embs_list = []
        pos_ohots_list = []
        sent_lens = []

        for caption in captions:
            words = caption.lower().split()
            tokens = ['sos/OTHER'] + [f'{word}/OTHER' for word in words] + ['eos/OTHER']
            sent_len = len(tokens)

            pos_one_hots_list = []
            word_embeddings_list = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots_list.append(pos_oh[None, :])
                word_embeddings_list.append(word_emb[None, :])

            pos_ohot = np.concatenate(pos_one_hots_list, axis=0)
            word_embs = np.concatenate(word_embeddings_list, axis=0)

            word_embs_list.append(torch.from_numpy(word_embs))
            pos_ohots_list.append(torch.from_numpy(pos_ohot))
            sent_lens.append(sent_len)

        max_len = max(sent_lens)
        word_dim = word_embs_list[0].shape[-1]
        pos_dim = pos_ohots_list[0].shape[-1]

        word_embeddings = torch.zeros(batch_size, max_len, word_dim)
        pos_one_hots = torch.zeros(batch_size, max_len, pos_dim)

        for i in range(batch_size):
            cur_len = sent_lens[i]
            word_embeddings[i, :cur_len] = word_embs_list[i]
            pos_one_hots[i, :cur_len] = pos_ohots_list[i]

        sent_lens = torch.tensor(sent_lens, device=self.device)
        return word_embeddings, pos_one_hots, sent_lens

    def _compute_matching_score(
        self,
        text_emb: torch.Tensor,
        motion_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE reward. Also populates self._reward_stats."""
        text_emb_norm = F.normalize(text_emb, p=2, dim=-1)
        motion_emb_norm = F.normalize(motion_emb, p=2, dim=-1)

        sim_matrix = motion_emb_norm @ text_emb_norm.T
        B = sim_matrix.shape[0]
        positive_sim = sim_matrix.diag()

        if B > 1:
            logits = sim_matrix / self.tau
            scores = logits.diag() - torch.logsumexp(logits, dim=-1)
            mask = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
            negative_sim = (sim_matrix * mask).sum(dim=-1) / (B - 1)
        else:
            scores = torch.zeros(1, device=sim_matrix.device)
            negative_sim = torch.zeros(1, device=sim_matrix.device)

        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=-10.0)

        self._reward_stats = {
            'pos_sim_mean': positive_sim.mean().item(),
            'neg_sim_mean': negative_sim.mean().item(),
        }

        return scores


# ---------------------------------------------------------------------------
# Utility: standalone test
# ---------------------------------------------------------------------------

def test_reward_model(reward_model: GRPORewardModel):
    """Quick sanity check."""
    print("Testing GRPO Reward Model...")

    captions = [
        "a person walks forward",
        "a person jumps up and down three times",
        "a person takes four steps forward",
    ]

    motion_tokens = [
        torch.randint(0, 512, (64,)),
        torch.randint(0, 512, (48,)),
        torch.randint(0, 512, (32,)),
    ]

    rewards, components = reward_model.compute_reward(
        captions, motion_tokens, return_components=True,
    )

    print(f"Rewards: {rewards}")
    print(f"Matching (cos sim): {components['matching_scores']}")
    print(f"Physical scores: {components['physical_scores']}")
    print(f"Numerical scores: {components['numerical_scores']}")
    print(f"Has numerical: {components['has_numerical']}")
    print(f"Stats: {components['reward_stats']}")
    print("Test passed!")
    return rewards, components
