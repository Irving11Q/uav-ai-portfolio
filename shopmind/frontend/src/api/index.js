import axios from 'axios'

const http = axios.create({ baseURL: '/api' })

http.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
    // 同时走自定义头：部署到云平台后，边缘网关可能注入/改写 Authorization，
    // 导致后端拿不到我们的 token（现象是登录成功但所有接口 401）。
    // 后端会优先读这个头，Authorization 退为兜底。
    config.headers['X-Auth-Token'] = token
  }
  return config
})

http.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
    }
    return Promise.reject(err)
  }
)

export function login(username, password) {
  return http.post('/auth/login', { username, password }).then((r) => r.data)
}
export function register(username, password) {
  return http.post('/auth/register', { username, password }).then((r) => r.data)
}
export function me() {
  return http.get('/auth/me').then((r) => r.data)
}
export function logout() {
  return http.post('/auth/logout').then((r) => r.data)
}
export function changePassword(old_password, new_password) {
  return http
    .post('/auth/change-password', { old_password, new_password })
    .then((r) => r.data)
}

// ---- 知识库管理（管理员）----
export function listDocuments() {
  return http.get('/documents').then((r) => r.data)
}
export function docStats() {
  return http.get('/documents/stats').then((r) => r.data)
}
export function uploadDocument(file) {
  const fd = new FormData()
  fd.append('file', file)
  return http.post('/documents', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data)
}
export function reindexDocument(id) {
  return http.post(`/documents/${id}/reindex`).then((r) => r.data)
}
export function deleteDocument(id) {
  return http.delete(`/documents/${id}`).then((r) => r.data)
}

// 下载原始文件（用户拿到的始终是上传时的原格式，如 .docx，而非内部 Markdown）
export function downloadDocument(id) {
  return http.get(`/documents/${id}/download`, { responseType: 'blob' })
}
// 检索/缓存/时延可观测指标（管理员）
export function queryStats() {
  return http.get('/stats').then((r) => r.data)
}

// ---- 会话与问答 ----
export function createSession(title) {
  return http.post('/chat/sessions', { title }).then((r) => r.data)
}
export function listSessions() {
  return http.get('/chat/sessions').then((r) => r.data)
}
export function deleteSession(id) {
  return http.delete(`/chat/sessions/${id}`).then((r) => r.data)
}
export function getMessages(id) {
  return http.get(`/chat/sessions/${id}/messages`).then((r) => r.data)
}
export function askQuestion(id, question) {
  return http.post(`/chat/sessions/${id}/ask`, { question }).then((r) => r.data)
}

/** 流式问答（SSE）：答案边生成边回调，不用干等整段生成完。
 *
 * 为什么用原生 fetch 而不是 axios：浏览器端 axios 走 XHR，拿不到流式响应体
 * （只能等 body 全部到达）。fetch + ReadableStream 才能逐块读。
 * 代价是绕过了 axios 拦截器，所以 token 要自己塞 —— 这里刻意和拦截器保持
 * 同一套头（Authorization + X-Auth-Token），免得线上网关改写时行为不一致。
 *
 * @param {number} sessionId
 * @param {string} question
 * @param {(ev: object) => void} onEvent  每个事件回调，ev.type ∈ meta|references|delta|done|final|error
 * @param {AbortSignal} [signal]          取消用
 */
export async function askQuestionStream(sessionId, question, onEvent, signal) {
  const token = localStorage.getItem('token') || ''
  const res = await fetch(`/api/chat/sessions/${sessionId}/ask/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      'X-Auth-Token': token,
    },
    body: JSON.stringify({ question }),
    signal,
  })

  if (!res.ok) {
    // 限流 429 / 未授权 401 等都在「开始推流之前」返回，所以这里拿得到正常的状态码
    let detail = `HTTP ${res.status}`
    try {
      const j = await res.json()
      if (j && j.detail) detail = j.detail
    } catch {
      /* 响应体不是 JSON，保留状态码文案 */
    }
    const err = new Error(detail)
    err.status = res.status
    throw err
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    // 去掉 \r：SSE 允许 CRLF 分帧，而 JSON 里的 \r 本来就是转义过的，不会有裸 \r
    buf = (buf + decoder.decode(value, { stream: true })).replace(/\r/g, '')
    let sep
    while ((sep = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      const line = frame.split('\n').find((l) => l.startsWith('data:'))
      if (!line) continue
      const payload = line.slice(5).trim()
      if (!payload) continue
      try {
        onEvent(JSON.parse(payload))
      } catch {
        /* 半帧 / 坏帧跳过，别让一次解析失败打断整条流 */
      }
    }
  }
}

/** 判断当前浏览器/环境是否支持流式读取（拿不到 ReadableStream 时上层可回退同步接口）。 */
export function supportsStreaming() {
  return typeof window !== 'undefined' && typeof window.ReadableStream !== 'undefined'
}

// ---- 阶段6：答案反馈 ----
export function feedbackReasons() {
  return http.get('/feedback/reasons').then((r) => r.data)
}
export function submitFeedback(messageId, rating, reason, comment) {
  return http
    .post('/feedback', { message_id: messageId, rating, reason, comment })
    .then((r) => r.data)
}
export function feedbackSummary() {
  return http.get('/feedback/summary').then((r) => r.data)
}
export function listBadCases(params) {
  return http.get('/feedback/bad-cases', { params }).then((r) => r.data)
}
export function resolveBadCase(id, note, resolved = true) {
  return http
    .post(`/feedback/${id}/resolve`, { resolved, note })
    .then((r) => r.data)
}

// ---- 阶段6：统计仪表盘 ----
export function dashboard() {
  return http.get('/dashboard').then((r) => r.data)
}
export function dashboardTrend(days = 7) {
  return http.get('/dashboard/trend', { params: { days } }).then((r) => r.data)
}
export function dashboardHotQuestions(limit = 10, days = 30) {
  return http.get('/dashboard/hot-questions', { params: { limit, days } }).then((r) => r.data)
}
export function dashboardDocuments() {
  return http.get('/dashboard/documents').then((r) => r.data)
}
export function dashboardActions() {
  return http.get('/dashboard/actions').then((r) => r.data)
}

// ---- 阶段6：热问榜与相似问题（任意登录用户）----
export function hotQuestions(limit = 8, days = 30) {
  return http.get('/chat/hot-questions', { params: { limit, days } }).then((r) => r.data)
}
export function similarQuestions(question, k = 3) {
  return http.get('/chat/similar-questions', { params: { question, k } }).then((r) => r.data)
}

// ---- 阶段6：对话导出（blob 下载）----
export function exportSession(id, format = 'md') {
  return http.get(`/chat/sessions/${id}/export`, {
    params: { format },
    responseType: 'blob',
  })
}

// ---- 演示模式额度（公开体验版）----
export function demoQuota() {
  return http.get('/demo/quota').then((r) => r.data)
}

// ---- 阶段6：运行时配置 ----
export function getSettings() {
  return http.get('/settings').then((r) => r.data)
}
export function publicSettings() {
  return http.get('/settings/public').then((r) => r.data)
}
export function updateSettings(values) {
  return http.put('/settings', { values }).then((r) => r.data)
}
export function resetSettings() {
  return http.post('/settings/reset').then((r) => r.data)
}

// ---- 阶段6：审计日志 ----
export function auditLogs(params) {
  return http.get('/audit-logs', { params }).then((r) => r.data)
}
export function auditActions() {
  return http.get('/audit-logs/actions').then((r) => r.data)
}
export function auditUsers() {
  return http.get('/audit-logs/users').then((r) => r.data)
}
export function exportAuditLogs(params) {
  return http.get('/audit-logs/export', { params, responseType: 'blob' })
}

export default http
