<template>
  <div class="auth-wrap">
    <div class="auth-card">
      <div class="auth-head">
        <div class="mark">RA</div>
        <h2 class="auth-title">欢迎回来</h2>
        <p class="auth-sub">电商 RAG 知识库问答系统</p>
      </div>

      <el-form @submit.prevent="onSubmit" size="large">
        <el-form-item>
          <el-input v-model="username" placeholder="用户名" clearable>
            <template #prefix><el-icon><User /></el-icon></template>
          </el-input>
        </el-form-item>
        <el-form-item>
          <el-input
            ref="pwdRef"
            v-model="password"
            type="password"
            show-password
            placeholder="密码"
            @keyup.enter="onSubmit"
          >
            <template #prefix><el-icon><Lock /></el-icon></template>
          </el-input>
        </el-form-item>
        <el-button
          type="primary"
          class="submit"
          :loading="loading"
          @click="onSubmit"
        >
          登 录
        </el-button>
      </el-form>

      <div class="auth-foot">
        <span class="muted">还没有账号？</span>
        <el-button link type="primary" @click="goRegister">立即注册</el-button>
      </div>

      <div class="auth-tip">
        管理员演示账号请见项目说明文档
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import { useAuthStore } from '@/store/auth'

// 从注册页跳回来时带 ?username=xxx，直接填好用户名，用户只需再输密码
const route = useRoute()
const username = ref(route.query.username || '')
const password = ref('')
const pwdRef = ref(null)
const loading = ref(false)
const router = useRouter()
const auth = useAuthStore()

onMounted(() => {
  if (username.value) pwdRef.value?.focus()
})

async function onSubmit() {
  if (!username.value || !password.value) {
    ElMessage.warning('请输入用户名和密码')
    return
  }
  loading.value = true
  try {
    await auth.login(username.value, password.value)
    ElMessage.success('登录成功')
    router.push(route.query.redirect || '/')
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '登录失败')
  } finally {
    loading.value = false
  }
}

// 注册走独立页面 /register，登录页只负责登录
function goRegister() {
  router.push({ name: 'register' })
}
</script>

<style scoped>
.auth-wrap {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--app-bg);
  padding: 24px;
}
.auth-card {
  width: 100%;
  max-width: 396px;
  background: var(--app-card);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius-lg);
  box-shadow: var(--app-shadow);
  padding: 34px 32px 26px;
}
.auth-head {
  text-align: center;
  margin-bottom: 24px;
}
.mark {
  width: 46px;
  height: 46px;
  border-radius: 14px;
  background: var(--app-primary);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  font-weight: 500;
  margin: 0 auto 14px;
  letter-spacing: 0.5px;
}
.auth-title {
  margin: 0;
  font-size: 20px;
  font-weight: 500;
}
.auth-sub {
  margin: 6px 0 0;
  color: var(--app-text-3);
  font-size: 12.5px;
}
.submit {
  width: 100%;
  height: 42px;
  margin-top: 4px;
}
.auth-foot {
  margin-top: 18px;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  font-size: 13px;
}
.auth-tip {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--app-border-soft);
  text-align: center;
  color: var(--app-text-3);
  font-size: 12px;
}
:deep(.el-form-item) {
  margin-bottom: 16px;
}
</style>
