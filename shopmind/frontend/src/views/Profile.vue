<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h2 class="page-title">账户设置</h2>
        <p class="page-sub">查看账号信息并修改登录密码</p>
      </div>
    </div>

    <div class="two-col">
      <!-- 左：账号信息（只读） -->
      <div class="app-card">
        <h3 class="card-title">账号信息</h3>

        <div class="info-list">
          <div class="info-row">
            <span class="info-label">用户名</span>
            <span class="info-value">{{ auth.user?.username || '-' }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">账号 ID</span>
            <span class="info-value">{{ auth.user?.id ?? '-' }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">角色</span>
            <span class="info-value">
              <span class="soft-tag" :class="{ gray: !auth.isAdmin }">{{ roleText }}</span>
            </span>
          </div>
          <div class="info-row">
            <span class="info-label">注册时间</span>
            <span class="info-value">{{ createdText }}</span>
          </div>
        </div>

        <!-- 权限用标签罗列，比一整句散文好扫读 -->
        <div class="perm-block">
          <div class="perm-title">权限范围</div>
          <div class="perm-list">
            <span v-for="p in perms" :key="p" class="perm-chip">{{ p }}</span>
          </div>
        </div>
      </div>

      <!-- 右：修改密码 -->
      <div class="app-card">
        <h3 class="card-title">修改密码</h3>

        <!-- label 放输入框上方：既不会被定宽截断，也让输入框更短更好读 -->
        <el-form label-position="top" class="pwd-form" @submit.prevent>
          <el-form-item label="原密码">
            <el-input
              v-model="oldPwd"
              type="password"
              show-password
              placeholder="请输入当前密码"
            />
          </el-form-item>
          <el-form-item label="新密码">
            <el-input
              v-model="newPwd"
              type="password"
              show-password
              placeholder="至少 6 位"
              @keyup.enter="onSubmit"
            />
          </el-form-item>
          <el-form-item label="确认新密码">
            <el-input
              v-model="confirmPwd"
              type="password"
              show-password
              placeholder="再输入一次新密码"
              @keyup.enter="onSubmit"
            />
          </el-form-item>

          <!-- 提示与按钮同行：按钮右边缘与输入框对齐，不再孤零零挂在左下角 -->
          <div class="pwd-actions">
            <span class="pwd-hint">建议字母 + 数字组合</span>
            <el-button type="primary" :loading="loading" @click="onSubmit">保存修改</el-button>
          </div>
        </el-form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/store/auth'
import { changePassword } from '@/api'

const auth = useAuthStore()
const oldPwd = ref('')
const newPwd = ref('')
const confirmPwd = ref('')
const loading = ref(false)

const roleText = computed(() => (auth.isAdmin ? '管理员' : '普通用户'))

// 后端 created_at 是 UTC 的无时区串，直接取日期部分展示，不做本地时区换算，
// 避免"看起来对了但其实偏了一天"这种难以察觉的口径问题。
const createdText = computed(() => {
  const s = auth.user?.created_at
  return s ? String(s).slice(0, 10) : '-'
})

// 与侧边栏实际可见的菜单保持一致，别写系统里不存在的功能
const perms = computed(() =>
  auth.isAdmin
    ? ['知识库问答', '知识库管理', '数据看板', '反馈与调优', '审计日志', '系统设置']
    : ['知识库问答', '个人账户设置']
)

async function onSubmit() {
  if (!oldPwd.value || !newPwd.value) {
    ElMessage.warning('请填写原密码与新密码')
    return
  }
  if (newPwd.value.length < 6) {
    ElMessage.warning('新密码至少 6 位')
    return
  }
  if (newPwd.value !== confirmPwd.value) {
    ElMessage.warning('两次输入的新密码不一致')
    return
  }
  loading.value = true
  try {
    await changePassword(oldPwd.value, newPwd.value)
    ElMessage.success('密码已修改')
    oldPwd.value = ''
    newPwd.value = ''
    confirmPwd.value = ''
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '修改失败')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.two-col {
  display: grid;
  /* 用 minmax(0, 1fr) 而不是 1fr：1fr 的最小值是 auto，
     表单控件的最小内容宽度会把整列撑破，导致两列不等宽。 */
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
/* 全局 `.app-card + .app-card { margin-top: 16px }` 是给纵向堆叠用的，
   在两列网格里会把第二张卡片整体压下去 16px（实测 148 vs 164，顶边不齐）。 */
.two-col > .app-card + .app-card {
  margin-top: 0;
}
@media (max-width: 900px) {
  .two-col {
    grid-template-columns: minmax(0, 1fr);
  }
  /* 单列时恢复纵向间距 */
  .two-col > .app-card + .app-card {
    margin-top: 16px;
  }
}

/* ---------- 左：账号信息 ---------- */
.info-list {
  margin: 0;
}
.info-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 11px 0;
  border-bottom: 1px dashed var(--app-border-soft);
}
.info-row:last-child {
  border-bottom: none;
}
.info-label {
  width: 72px;
  flex: 0 0 72px;
  color: var(--app-text-2);
  font-size: 13px;
}
.info-value {
  font-size: 13.5px;
  line-height: 1.6;
  min-width: 0;
  word-break: break-word;
}

.perm-block {
  margin-top: 10px;
  padding-top: 14px;
  border-top: 1px solid var(--app-border-soft);
}
.perm-title {
  color: var(--app-text-2);
  font-size: 13px;
  margin-bottom: 9px;
}
.perm-list {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}
.perm-chip {
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  line-height: 1.5;
  color: var(--app-text-2);
  background: #f6f9f8;
  border: 1px solid var(--app-border);
}

/* ---------- 右：修改密码 ---------- */
.pwd-form :deep(.el-form-item) {
  margin-bottom: 14px;
}
.pwd-form :deep(.el-form-item__label) {
  padding-bottom: 5px;
  font-size: 13px;
  line-height: 1.4;
  color: var(--app-text-2);
}
.pwd-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 18px;
}
.pwd-hint {
  color: var(--app-text-3);
  font-size: 12px;
}
@media (max-width: 900px) {
  .pwd-actions {
    flex-direction: column-reverse;
    align-items: stretch;
  }
  .pwd-actions .el-button {
    width: 100%;
  }
  .pwd-hint {
    text-align: center;
  }
}
</style>
