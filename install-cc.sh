#!/bin/bash
set -e

echo "============================================"
echo "  Claude Code 一键安装脚本 (AutoDL 国内版)"
echo "============================================"

# ---------- 1. 开启网络代理 (访问 GitHub/npm) ----------
if [ -f /etc/network_turbo ]; then
    echo "[1/4] 开启学术网络加速..."
    source /etc/network_turbo
else
    echo "[1/4] 未检测到 /etc/network_turbo，跳过代理设置"
fi

# ---------- 2. 安装 Node.js 20.x (使用国内镜像) ----------
NODE_VERSION="v20.18.3"
ARCH="x64"
NODE_DIR="node-${NODE_VERSION}-linux-${ARCH}"
NODE_TAR="${NODE_DIR}.tar.xz"
INSTALL_DIR="/usr/local/nodejs"

# 先把已有安装路径加入 PATH，避免重复下载
if[ -d "${INSTALL_DIR}/bin" ]; then
    export PATH="${INSTALL_DIR}/bin:$PATH"
fi

if command -v node &>/dev/null && [[ "$(node -v)" == v2* || "$(node -v)" == v1[89]* ]]; then
    echo "[2/4] Node.js $(node -v) 已安装，跳过"
else
    echo "[2/4] 安装 Node.js ${NODE_VERSION} (npmmirror 国内镜像)..."
    NODE_URL="https://npmmirror.com/mirrors/node/${NODE_VERSION}/${NODE_TAR}"
    (
        cd /tmp
        echo "  下载中: ${NODE_URL}"
        # 加上 --insecure 防止 curl 报证书错误
        curl -fSL --insecure --progress-bar -o "${NODE_TAR}" "${NODE_URL}"

        echo "  解压中..."
        tar -xf "${NODE_TAR}"
        rm -f "${NODE_TAR}"

        rm -rf "${INSTALL_DIR}"
        mv "${NODE_DIR}" "${INSTALL_DIR}"
    )
    export PATH="/usr/local/nodejs/bin:$PATH"
    echo "  Node.js $(node -v) 安装完成"
    echo "  npm $(npm -v)"
fi

export PATH="/usr/local/nodejs/bin:$PATH"

# 持久化 PATH
cat > /etc/profile.d/nodejs.sh <<'ENVEOF'
export PATH="/usr/local/nodejs/bin:$PATH"
ENVEOF
if ! grep -qF '/usr/local/nodejs/bin' "${HOME}/.bashrc" 2>/dev/null; then
    echo 'export PATH="/usr/local/nodejs/bin:$PATH"' >> "${HOME}/.bashrc"
fi

# ---------- 3. 配置 npm 国内镜像 ----------
echo "[3/4] 配置 npm 国内镜像 (npmmirror)..."
npm config set registry https://registry.npmmirror.com
npm config set strict-ssl false
export NODE_TLS_REJECT_UNAUTHORIZED=0

# ---------- 4. 安装 Claude Code ----------
echo "[4/4] 安装 Claude Code..."
# 直接在命令末尾加上终极忽略证书参数，击穿 AutoDL 的代理拦截
npm install -g @anthropic-ai/claude-code --strict-ssl=false

NPM_GLOBAL_BIN="$(npm prefix -g)/bin"
export PATH="${NPM_GLOBAL_BIN}:$PATH"

echo ""
echo "============================================"
echo "  安装完成!"
echo "============================================"
echo ""
echo "  Node.js: $(node -v)"
echo "  npm:     $(npm -v)"
echo "  Claude:  $(claude --version 2>/dev/null || echo '请重新打开终端后运行 claude --version')"
echo "============================================"
