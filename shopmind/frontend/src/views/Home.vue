<template>
  <div class="shell">
    <!-- 侧边导航：浅色、圆角选中态、分组标签 -->
    <aside class="side">
      <div class="brand">
        <div class="brand-mark">RA</div>
        <div class="brand-text">
          <div class="brand-name">RAG 知识库</div>
          <div class="brand-sub">电商商品问答</div>
        </div>
      </div>

      <nav class="nav">
        <div class="nav-group">
          <el-menu :default-active="$route.path" router class="nav-menu">
            <el-menu-item index="/chat">
              <el-icon><ChatDotRound /></el-icon>
              <span>知识库问答</span>
            </el-menu-item>

            <template v-if="auth.isAdmin">
              <div class="nav-label">管理</div>
              <el-menu-item index="/dashboard">
                <el-icon><DataAnalysis /></el-icon>
                <span>数据看板</span>
              </el-menu-item>
              <el-menu-item index="/kb">
                <el-icon><FolderOpened /></el-icon>
                <span>知识库管理</span>
              </el-menu-item>
              <el-menu-item index="/feedback">
                <el-icon><Tickets /></el-icon>
                <span>反馈与调优</span>
              </el-menu-item>
              <el-menu-item index="/audit">
                <el-icon><Memo /></el-icon>
                <span>审计日志</span>
              </el-menu-item>
              <el-menu-item index="/settings">
                <el-icon><Setting /></el-icon>
                <span>系统设置</span>
              </el-menu-item>
            </template>

            <div class="nav-label">个人</div>
            <el-menu-item index="/profile">
              <el-icon><User /></el-icon>
              <span>账户设置</span>
            </el-menu-item>
          </el-menu>
        </div>
      </nav>
    </aside>

    <!-- 主区：顶栏 + 内容 -->
    <div class="main">
      <header class="top">
        <div class="top-left">
          <span class="top-title">{{ pageTitle }}</span>
        </div>
        <div class="top-right">
          <span v-if="modelName" class="soft-tag">
            <el-icon><Cpu /></el-icon>模型 {{ modelName }}
          </span>
          <el-dropdown trigger="click" @command="onCommand">
            <span class="user-chip">
              <span class="avatar">{{ (auth.user?.username || '?').slice(0, 1).toUpperCase() }}</span>
              <span class="user-meta">
                <span class="user-name">{{ auth.user?.username || '未登录' }}</span>
                <span class="user-role">{{ auth.isAdmin ? '管理员' : '普通用户' }}</span>
              </span>
              <el-icon class="chev"><ArrowDown /></el-icon>
            </span>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="profile">账户设置</el-dropdown-item>
                <el-dropdown-item command="logout" divided>退出登录</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </header>

      <main class="content">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  ChatDotRound,
  DataAnalysis,
  FolderOpened,
  Tickets,
  Memo,
  Setting,
  User,
  Cpu,
  ArrowDown,
} from '@element-plus/icons-vue'
import { useAuthStore } from '@/store/auth'
import { publicSettings } from '@/api'

const auth = useAuthStore()
const router = useRouter()
const route = useRoute()
const modelName = ref('')

const pageTitle = computed(() => route.meta?.title || '知识库问答')

onMounted(async () => {
  if (!auth.user) auth.fetchMe()
  // 顶栏展示当前生效的模型名（非敏感配置，普通用户也能看）
  try {
    const s = await publicSettings()
    modelName.value = s.LLM_MODEL || ''
  } catch {
    modelName.value = ''
  }
})

async function onCommand(cmd) {
  if (cmd === 'logout') {
    // 等服务端清掉 cookie 再跳转，避免带着残留登录态回到登录页
    await auth.logout()
    router.push('/login')
  } else if (cmd === 'profile') {
    router.push('/profile')
  }
}
</script>

<style scoped>
.shell {
  display: flex;
  height: 100vh;
  background: var(--app-bg);
}

/* ---------- 侧边栏 ---------- */
.side {
  width: 216px;
  flex: 0 0 216px;
  background: var(--app-card);
  border-right: 1px solid var(--app-border);
  display: flex;
  flex-direction: column;
  padding: 16px 12px;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 8px 18px;
}
.brand-mark {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: var(--app-primary);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.5px;
}
.brand-name {
  font-size: 14.5px;
  font-weight: 500;
  line-height: 1.3;
}
.brand-sub {
  font-size: 11.5px;
  color: var(--app-text-3);
}

.nav {
  flex: 1;
  overflow-y: auto;
}
.nav-label {
  padding: 14px 12px 6px;
  font-size: 11.5px;
  color: var(--app-text-3);
  letter-spacing: 0.5px;
}
.nav-menu {
  border: none;
  background: transparent;
}
/* 菜单项做成圆角药丸，选中态用主色浅底 + 主色字 */
.nav-menu :deep(.el-menu-item) {
  height: 40px;
  line-height: 40px;
  border-radius: 10px;
  margin-bottom: 2px;
  color: var(--app-text-2);
  font-size: 13.5px;
  padding-left: 12px !important;
}
.nav-menu :deep(.el-menu-item:hover) {
  background: var(--app-border-soft);
  color: var(--app-text);
}
.nav-menu :deep(.el-menu-item.is-active) {
  background: var(--app-primary-soft);
  color: var(--app-primary-dark);
  font-weight: 500;
}
.nav-menu :deep(.el-menu-item .el-icon) {
  margin-right: 9px;
  font-size: 16px;
}

/* ---------- 顶栏 ---------- */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.top {
  height: 58px;
  flex: 0 0 58px;
  background: var(--app-card);
  border-bottom: 1px solid var(--app-border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 22px;
}
.top-title {
  font-size: 15px;
  font-weight: 500;
}
.top-right {
  display: flex;
  align-items: center;
  gap: 14px;
}
.user-chip {
  display: flex;
  align-items: center;
  gap: 9px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 10px;
  outline: none;
}
.user-chip:hover {
  background: var(--app-border-soft);
}
.avatar {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: var(--app-primary-soft);
  color: var(--app-primary-dark);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 500;
}
.user-meta {
  display: flex;
  flex-direction: column;
  line-height: 1.25;
}
.user-name {
  font-size: 13px;
}
.user-role {
  font-size: 11px;
  color: var(--app-text-3);
}
.chev {
  color: var(--app-text-3);
  font-size: 12px;
}

/* ---------- 内容区 ---------- */
.content {
  flex: 1;
  overflow-y: auto;
  padding: 20px 22px;
}
</style>
