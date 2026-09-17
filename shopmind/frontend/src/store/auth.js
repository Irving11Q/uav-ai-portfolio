import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { login as apiLogin, me as apiMe, logout as apiLogout } from '@/api'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('token') || '')
  const user = ref(null)
  const isAdmin = computed(() => user.value?.role === 'admin')

  function setToken(t) {
    token.value = t
    localStorage.setItem('token', t)
  }

  async function login(username, password) {
    const data = await apiLogin(username, password)
    setToken(data.access_token)
    await fetchMe()
  }

  async function fetchMe() {
    if (!token.value) return
    user.value = await apiMe()
  }

  /** 清掉本地登录态（token + 用户信息 + localStorage）。
   *  单独抽出来是因为路由守卫也要用：token 失效时要能就地清干净再跳登录页。 */
  function clearSession() {
    token.value = ''
    user.value = null
    localStorage.removeItem('token')
  }

  async function logout() {
    // 先请后端清掉 access_token cookie（登录时写的那份），
    // 否则 localStorage 清了、cookie 还在，服务端仍认为已登录。
    // 失败也要继续清本地状态 —— 退出登录不该被网络问题卡住。
    try {
      await apiLogout()
    } catch {
      /* 忽略 */
    }
    clearSession()
  }

  return { token, user, isAdmin, setToken, login, fetchMe, logout, clearSession }
})
