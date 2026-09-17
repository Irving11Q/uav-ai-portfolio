import { createRouter, createWebHistory } from 'vue-router'
import Login from '@/views/Login.vue'
import Register from '@/views/Register.vue'
import Home from '@/views/Home.vue'
import Chat from '@/views/Chat.vue'
import Dashboard from '@/views/Dashboard.vue'
import KnowledgeBase from '@/views/KnowledgeBase.vue'
import Feedback from '@/views/Feedback.vue'
import AuditLog from '@/views/AuditLog.vue'
import Settings from '@/views/Settings.vue'
import Profile from '@/views/Profile.vue'
import { useAuthStore } from '@/store/auth'

const routes = [
  { path: '/login', name: 'login', component: Login, meta: { public: true } },
  { path: '/register', name: 'register', component: Register, meta: { public: true } },
  {
    path: '/',
    component: Home,
    meta: { requiresAuth: true },
    children: [
      { path: '', redirect: '/chat' },
      {
        path: 'chat',
        name: 'chat',
        component: Chat,
        meta: { requiresAuth: true, title: '知识库问答' },
      },
      {
        path: 'dashboard',
        name: 'dashboard',
        component: Dashboard,
        meta: { requiresAuth: true, requiresAdmin: true, title: '数据看板' },
      },
      {
        path: 'kb',
        name: 'kb',
        component: KnowledgeBase,
        meta: { requiresAuth: true, requiresAdmin: true, title: '知识库管理' },
      },
      {
        path: 'feedback',
        name: 'feedback',
        component: Feedback,
        meta: { requiresAuth: true, requiresAdmin: true, title: '反馈与调优' },
      },
      {
        path: 'audit',
        name: 'audit',
        component: AuditLog,
        meta: { requiresAuth: true, requiresAdmin: true, title: '审计日志' },
      },
      {
        path: 'settings',
        name: 'settings',
        component: Settings,
        meta: { requiresAuth: true, requiresAdmin: true, title: '系统设置' },
      },
      {
        path: 'profile',
        name: 'profile',
        component: Profile,
        meta: { requiresAuth: true, title: '账户设置' },
      },
    ],
  },
  { path: '/:pathMatch(.*)*', redirect: '/' },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach(async (to) => {
  const auth = useAuthStore()
  if (to.meta.public) return true
  if (!auth.token) return { name: 'login', query: { redirect: to.fullPath } }

  // 整页刷新（或在地址栏直接输网址）后 Pinia 是空的：fetchMe 原本要等 Home 挂载
  // 之后才跑，晚于本守卫，于是 requiresAdmin 会误判成「无权限」，把管理员从
  // /dashboard、/kb 这类页面直接踢回问答页。这里先补齐用户信息再判定。
  if (!auth.user) {
    try {
      await auth.fetchMe()
    } catch {
      // 拉不到就当作登录已失效，清干净再跳登录页（别留在空白页）
      auth.clearSession()
      return { name: 'login', query: { redirect: to.fullPath } }
    }
  }

  if (to.meta.requiresAdmin && !auth.isAdmin) return { name: 'chat' }
  return true
})

export default router
