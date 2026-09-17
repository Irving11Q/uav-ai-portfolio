<template>
  <div class="auth-wrap">
    <div class="auth-card">
      <div class="auth-head">
        <div class="mark">RA</div>
        <h2 class="auth-title">创建账号</h2>
        <p class="auth-sub">注册后即可开始商品知识问答</p>
      </div>

      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        @submit.prevent="onSubmit"
        size="large"
      >
        <el-form-item prop="username">
          <el-input v-model="form.username" placeholder="用户名（3 到 32 位）" clearable>
            <template #prefix><el-icon><User /></el-icon></template>
          </el-input>
        </el-form-item>
        <el-form-item prop="password">
          <el-input
            v-model="form.password"
            type="password"
            show-password
            placeholder="密码（至少 6 位）"
          >
            <template #prefix><el-icon><Lock /></el-icon></template>
          </el-input>
        </el-form-item>
        <el-form-item prop="confirm">
          <el-input
            v-model="form.confirm"
            type="password"
            show-password
            placeholder="确认密码"
            @keyup.enter="onSubmit"
          >
            <template #prefix><el-icon><Lock /></el-icon></template>
          </el-input>
        </el-form-item>
        <el-button type="primary" class="submit" :loading="loading" @click="onSubmit">
          注 册
        </el-button>
      </el-form>

      <div class="auth-foot">
        <span class="muted">已有账号？</span>
        <el-button link type="primary" @click="goLogin">返回登录</el-button>
      </div>
      <div class="auth-tip">注册成功后会自动返回登录页，用户名已为你填好</div>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import { register } from '@/api'

const router = useRouter()
const formRef = ref(null)
const loading = ref(false)

const form = reactive({
  username: '',
  password: '',
  confirm: '',
})

// 两次密码一致性校验：必须放在 form 之后，validator 里才能读到 form.password
const rules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 32, message: '用户名长度 3 到 32 位', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 64, message: '密码长度 6 到 64 位', trigger: 'blur' },
  ],
  confirm: [
    { required: true, message: '请再次输入密码', trigger: 'blur' },
    {
      validator: (rule, value, callback) => {
        if (value !== form.password) callback(new Error('两次输入的密码不一致'))
        else callback()
      },
      trigger: 'blur',
    },
  ],
}

async function onSubmit() {
  if (!formRef.value) return
  try {
    await formRef.value.validate()
  } catch {
    return
  }

  loading.value = true
  const name = form.username.trim()
  try {
    await register(name, form.password)
    ElMessage.success('注册成功，请登录')
    // 带着用户名回登录页，登录页会自动填好，只需再输密码
    router.push({ name: 'login', query: { username: name } })
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '注册失败')
  } finally {
    loading.value = false
  }
}

function goLogin() {
  router.push({ name: 'login' })
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
