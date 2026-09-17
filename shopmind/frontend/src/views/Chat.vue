<template>
  <div class="chat-root">
    <!-- 左：会话列表 -->
    <aside class="sess-bar">
      <el-button type="primary" class="new-btn" :icon="Plus" @click="onNewSession">
        新建会话
      </el-button>
      <div class="sess-label">历史会话</div>
      <el-scrollbar class="sess-list">
        <div
          v-for="s in sessions"
          :key="s.id"
          class="sess-item"
          :class="{ active: s.id === currentId }"
          @click="onSelect(s.id)"
        >
          <el-icon class="sess-icon"><ChatLineSquare /></el-icon>
          <span class="sess-title" :title="s.title">{{ s.title || '新会话' }}</span>
          <el-icon class="sess-del" @click.stop="onDeleteSession(s.id)"><Delete /></el-icon>
        </div>
        <p v-if="!sessions.length" class="muted sess-empty">还没有会话，点上方新建</p>
      </el-scrollbar>
    </aside>

    <!-- 中：对话区 -->
    <section class="conv">
      <header class="conv-head">
        <div class="conv-title">
          <span>{{ currentSession?.title || '知识库问答' }}</span>
          <span v-if="sessionCached" class="soft-tag gray">本条命中缓存</span>
        </div>
        <div class="conv-actions">
          <!-- 演示模式才显示：公开体验站的额度提示，避免额度用完后体验者一头雾水 -->
          <span v-if="quota.enabled" class="soft-tag" :title="`公开演示环境共 ${quota.limit} 次大模型调用额度`">
            演示额度剩余 {{ quota.remaining }} 次
          </span>
          <el-dropdown :disabled="!currentId" @command="onExport">
            <el-button size="small" :icon="Download" :disabled="!currentId">
              导出对话<el-icon class="el-icon--right"><ArrowDown /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="md">导出 Markdown</el-dropdown-item>
                <el-dropdown-item command="pdf">导出 PDF</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </header>

      <el-scrollbar ref="scrollRef" class="msg-area">
        <template v-if="currentId">
          <div v-for="m in messages" :key="m.key" class="msg-row" :class="m.role">
            <div class="msg-bubble" :class="m.role">
              <div
                v-if="m.role === 'assistant'"
                class="msg-content md-body"
                v-html="renderRich(m.content)"
              ></div>
              <div v-else class="msg-content">{{ m.content }}</div>

              <!-- 引用片段 -->
              <div v-if="m.references && m.references.length" class="refs">
                <div class="refs-title">
                  <el-icon><DocumentCopy /></el-icon>引用知识库片段
                </div>
                <div v-for="(r, ri) in m.references" :key="ri" class="ref-card">
                  <div class="ref-head">
                    <span class="ref-src">{{ r.source }}</span>
                    <span class="ref-score">相关度 {{ r.score }}</span>
                  </div>
                  <div class="ref-snip md-body" v-html="renderRich(r.snippet)"></div>
                </div>
              </div>

              <!-- 反馈按钮（仅助手消息） -->
              <div v-if="m.role === 'assistant' && m.id" class="fb-bar">
                <span class="muted">这条回答有帮助吗？</span>
                <el-button
                  size="small"
                  :type="m.feedback === 'up' ? 'primary' : 'default'"
                  :icon="StarFilled"
                  round
                  @click="rate(m, 'up')"
                >
                  有帮助
                </el-button>
                <el-button
                  size="small"
                  :type="m.feedback === 'down' ? 'danger' : 'default'"
                  :icon="Star"
                  round
                  @click="openDownDialog(m)"
                >
                  没帮助
                </el-button>
                <span v-if="m.feedback" class="muted">已记录，谢谢反馈</span>
                <span v-if="m.feedback_reason" class="soft-tag gray">{{ m.feedback_reason }}</span>
              </div>
            </div>
          </div>

          <!-- 相似问题推荐 -->
          <div v-if="similar.length" class="similar">
            <span class="muted">你可能还想问：</span>
            <span
              v-for="(s, i) in similar"
              :key="i"
              class="similar-chip"
              @click="askDirectly(s.question)"
            >
              {{ s.question }}
            </span>
          </div>

          <div v-if="sending && !streamStarted" class="thinking">
            <span class="dot"></span>正在检索知识库并生成答案…
          </div>
        </template>

        <div v-else class="empty-conv">
          <div class="empty-icon"><el-icon><ChatDotRound /></el-icon></div>
          <h3>开始一次商品知识问答</h3>
          <p class="muted">选择左侧会话，或直接点下面任一常见问题开始</p>
          <div class="hot-inline">
            <span
              v-for="(h, i) in hot"
              :key="i"
              class="similar-chip"
              @click="askDirectly(h.question)"
            >
              {{ h.question }}
            </span>
          </div>
        </div>
      </el-scrollbar>

      <footer class="input-bar">
        <el-input
          v-model="question"
          type="textarea"
          :rows="2"
          resize="none"
          placeholder="输入商品相关问题，回车发送（Shift + 回车换行）"
          @keydown.enter.exact.prevent="onSend"
        />
        <el-button type="primary" :loading="sending" :icon="Promotion" @click="onSend">
          发送
        </el-button>
      </footer>
    </section>

    <!-- 右：热问榜 -->
    <aside class="side-panel">
      <h4 class="side-title">热问榜</h4>
      <p class="muted side-hint">近 30 天大家问得最多的问题</p>
      <div v-if="hot.length" class="hot-list">
        <div
          v-for="(h, i) in hot"
          :key="i"
          class="hot-item"
          @click="askDirectly(h.question)"
        >
          <span class="hot-rank" :class="{ top: i < 3 }">{{ i + 1 }}</span>
          <span class="hot-q" :title="h.question">{{ h.question }}</span>
          <span class="hot-count">{{ h.count }}</span>
        </div>
      </div>
      <p v-else class="muted">还没有提问记录</p>
    </aside>

    <!-- 「没帮助」原因弹窗 -->
    <el-dialog v-model="downDialog" title="告诉我们哪里不对" width="440px">
      <p class="muted" style="margin-top: 0">
        你的反馈会进入 bad case 列表，用来改进知识库与检索效果。
      </p>
      <el-radio-group v-model="downReason" class="reason-group">
        <el-radio v-for="r in reasons" :key="r" :label="r" border>{{ r }}</el-radio>
      </el-radio-group>
      <el-input
        v-model="downComment"
        type="textarea"
        :rows="3"
        placeholder="补充说明（选填）"
        style="margin-top: 12px"
      />
      <template #footer>
        <el-button @click="downDialog = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="submitDown">提交反馈</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Plus,
  Delete,
  ChatLineSquare,
  ChatDotRound,
  DocumentCopy,
  Download,
  ArrowDown,
  Promotion,
  Star,
  StarFilled,
} from '@element-plus/icons-vue'
import {
  createSession,
  listSessions,
  deleteSession,
  getMessages,
  askQuestion,
  askQuestionStream,
  supportsStreaming,
  exportSession,
  similarQuestions,
  hotQuestions,
  feedbackReasons,
  submitFeedback,
  demoQuota,
} from '@/api'
import { renderRich } from '@/utils/markdown'

const sessions = ref([])
const currentId = ref(null)
const messages = ref([])
const question = ref('')
const sending = ref(false)
const scrollRef = ref(null)
const similar = ref([])
const hot = ref([])
const reasons = ref([])
const sessionCached = ref(false)
// 流式：首个字到达前显示「检索中…」，到达后就让文字自己长，提示得撤掉
const streamStarted = ref(false)
// 演示模式额度（非演示模式 enabled=false，界面上什么都不显示）
const quota = ref({ enabled: false, limit: 0, used: 0, remaining: -1 })

const downDialog = ref(false)
const downReason = ref('')
const downComment = ref('')
const downTarget = ref(null)
const submitting = ref(false)

let keySeq = 0
const nextKey = () => `m${++keySeq}`

const currentSession = computed(() =>
  sessions.value.find((s) => s.id === currentId.value)
)

async function refreshSessions() {
  try {
    sessions.value = await listSessions()
    if (!currentId.value && sessions.value.length) {
      await onSelect(sessions.value[0].id)
    }
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '加载会话失败')
  }
}

async function refreshQuota() {
  try {
    quota.value = await demoQuota()
  } catch {
    // 拿不到额度信息不影响问答，保持默认（不显示）
  }
}

async function onNewSession() {
  try {
    const s = await createSession('商品咨询')
    await refreshSessions()
    await onSelect(s.id)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '新建会话失败')
  }
}

async function onSelect(id) {
  currentId.value = id
  similar.value = []
  try {
    const list = await getMessages(id)
    messages.value = list.map((m) => ({ ...m, key: nextKey() }))
    scrollBottom()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '加载消息失败')
  }
}

async function onDeleteSession(id) {
  try {
    await deleteSession(id)
    if (currentId.value === id) currentId.value = null
    await refreshSessions()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '删除失败')
  }
}

/** 直接提问（点热问榜 / 相似问题 chip） */
async function askDirectly(text) {
  question.value = text
  await onSend()
}

async function onSend() {
  const q = question.value.trim()
  if (!q || sending.value) return
  if (!currentId.value) await onNewSession()
  if (!currentId.value) return

  messages.value.push({ key: nextKey(), role: 'user', content: q, references: [], id: null })
  question.value = ''
  similar.value = []
  sending.value = true
  streamStarted.value = false
  scrollBottom()

  // 先把助手气泡占位推进去，后续的增量都往这个气泡里追加。
  // ⚠️ 必须通过 messages.value[idx] 访问（那是 Vue 的响应式代理）——
  // 直接改闭包里持有的那个原对象，属性虽然变了但**不会触发视图更新**。
  messages.value.push({
    key: nextKey(),
    id: null,
    role: 'assistant',
    content: '',
    references: [],
    feedback: null,
  })
  const idx = messages.value.length - 1
  const bubble = () => messages.value[idx]

  try {
    if (supportsStreaming()) {
      await askQuestionStream(currentId.value, q, (ev) => {
        if (ev.type === 'delta') {
          streamStarted.value = true
          bubble().content += ev.text
          scrollBottom()
        } else if (ev.type === 'references') {
          // 引用是「检索完立即推」的，比第一个字还早，用户先看到溯源卡片
          bubble().references = ev.references || []
          scrollBottom()
        } else if (ev.type === 'done') {
          sessionCached.value = !!ev.cached
        } else if (ev.type === 'final') {
          // 落库成功后才拿得到 message_id，反馈按钮要靠它
          bubble().id = ev.message_id
        } else if (ev.type === 'error') {
          bubble().content = '调用失败：' + ev.message
          bubble().references = []
        }
      })
    } else {
      // 极老环境没有 ReadableStream：退回同步接口 —— 功能一点不缺，只是没有逐字效果
      const res = await askQuestion(currentId.value, q)
      bubble().content = res.answer
      bubble().references = res.references || []
      bubble().id = res.message_id
      sessionCached.value = !!res.cached
    }

    if (!bubble().content) bubble().content = '（模型没有返回内容，请重试）'
    scrollBottom()
    await refreshSessions()
    loadSimilar(q)
    refreshQuota()
  } catch (e) {
    // 走到这里说明「流还没开始就失败了」（429 限流 / 401 / 网络错误）。
    // 已经流到一半才出错的话，错误是以 error 事件送回来的，不在这里重复提示。
    const detail = e?.message || '问答失败'
    ElMessage.error(detail)
    if (!bubble().content) bubble().content = '调用失败：' + detail
  } finally {
    sending.value = false
    streamStarted.value = false
  }
}

async function loadSimilar(q) {
  try {
    similar.value = await similarQuestions(q, 3)
  } catch {
    similar.value = []
  }
}

async function rate(m, rating) {
  if (!m.id) return
  try {
    await submitFeedback(m.id, rating, null, null)
    m.feedback = rating
    m.feedback_reason = null
    ElMessage.success(rating === 'up' ? '已记录，谢谢反馈' : '已记录')
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '提交反馈失败')
  }
}

function openDownDialog(m) {
  if (!m.id) return
  downTarget.value = m
  downReason.value = m.feedback_reason || ''
  downComment.value = ''
  downDialog.value = true
}

async function submitDown() {
  if (!downTarget.value) return
  submitting.value = true
  try {
    await submitFeedback(downTarget.value.id, 'down', downReason.value || null, downComment.value || null)
    downTarget.value.feedback = 'down'
    downTarget.value.feedback_reason = downReason.value || null
    downDialog.value = false
    ElMessage.success('反馈已提交，会进入 bad case 列表用于调优')
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '提交失败')
  } finally {
    submitting.value = false
  }
}

async function onExport(fmt) {
  if (!currentId.value) return
  try {
    const res = await exportSession(currentId.value, fmt)
    const blob = new Blob([res.data], {
      type: fmt === 'pdf' ? 'application/pdf' : 'text/markdown;charset=utf-8',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${(currentSession.value?.title || '对话').slice(0, 20)}.${fmt}`
    a.click()
    URL.revokeObjectURL(url)
    ElMessage.success(`已导出 ${fmt.toUpperCase()}`)
  } catch (e) {
    ElMessage.error('导出失败，可能是该会话还没有内容')
  }
}

function scrollBottom() {
  nextTick(() => {
    const el = scrollRef.value?.wrapRef
    if (el) el.scrollTop = el.scrollHeight
  })
}

onMounted(async () => {
  await refreshSessions()
  try {
    hot.value = await hotQuestions(8, 30)
  } catch {
    hot.value = []
  }
  try {
    reasons.value = await feedbackReasons()
  } catch {
    reasons.value = []
  }
  await refreshQuota()
})
</script>

<style scoped>
.chat-root {
  display: flex;
  gap: 16px;
  height: calc(100vh - 58px - 40px);
  min-height: 480px;
}

/* ---------- 会话列表 ---------- */
.sess-bar {
  width: 222px;
  flex: 0 0 222px;
  background: var(--app-card);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  padding: 12px;
  display: flex;
  flex-direction: column;
  box-shadow: var(--app-shadow);
}
.new-btn {
  width: 100%;
}
.sess-label {
  margin: 14px 4px 6px;
  font-size: 11.5px;
  color: var(--app-text-3);
}
.sess-list {
  flex: 1;
}
.sess-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 10px;
  border-radius: 10px;
  cursor: pointer;
  color: var(--app-text-2);
}
.sess-item:hover {
  background: var(--app-border-soft);
}
.sess-item.active {
  background: var(--app-primary-soft);
  color: var(--app-primary-dark);
}
.sess-icon {
  flex: 0 0 auto;
  font-size: 15px;
}
.sess-title {
  flex: 1;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.sess-del {
  flex: 0 0 auto;
  opacity: 0;
  font-size: 14px;
  color: var(--app-text-3);
}
.sess-item:hover .sess-del {
  opacity: 1;
}
.sess-del:hover {
  color: var(--app-down);
}
.sess-empty {
  padding: 10px 4px;
}

/* ---------- 对话区 ---------- */
.conv {
  flex: 1;
  min-width: 0;
  background: var(--app-card);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  display: flex;
  flex-direction: column;
  box-shadow: var(--app-shadow);
}
.conv-head {
  height: 52px;
  flex: 0 0 52px;
  border-bottom: 1px solid var(--app-border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
}
.conv-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 500;
}
/* 演示额度提示 + 导出按钮同行对齐 */
.conv-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
.msg-area {
  flex: 1;
  padding: 18px 18px 8px;
}
.msg-row {
  display: flex;
  margin-bottom: 16px;
}
.msg-row.user {
  justify-content: flex-end;
}
.msg-bubble {
  max-width: 78%;
  padding: 11px 15px;
  border-radius: 14px;
  background: #f7faf9;
  border: 1px solid var(--app-border-soft);
}
.msg-bubble.user {
  background: var(--app-primary);
  border-color: var(--app-primary);
  color: #fff;
}
.msg-content {
  font-size: 13.5px;
  line-height: 1.7;
  word-break: break-word;
}

/* 引用卡片 */
.refs {
  margin-top: 12px;
  border-top: 1px dashed var(--app-border);
  padding-top: 10px;
}
.refs-title {
  font-size: 12px;
  color: var(--app-text-2);
  display: flex;
  align-items: center;
  gap: 5px;
  margin-bottom: 7px;
}
.ref-card {
  background: #fff;
  border: 1px solid var(--app-border);
  border-radius: 10px;
  padding: 8px 10px;
  margin-top: 6px;
}
.ref-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.ref-src {
  font-size: 12px;
  font-weight: 500;
}
.ref-score {
  font-size: 11.5px;
  color: var(--app-text-3);
}
.ref-snip {
  color: var(--app-text-2);
  margin-top: 4px;
  max-height: 180px;
  overflow: auto;
  font-size: 12.5px;
}

/* 反馈条 */
.fb-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 11px;
  padding-top: 9px;
  border-top: 1px dashed var(--app-border-soft);
  font-size: 12.5px;
}

/* 相似问题 */
.similar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 4px 2px 14px;
}
.similar-chip {
  padding: 5px 12px;
  border-radius: 999px;
  background: var(--app-primary-soft);
  color: var(--app-primary-dark);
  border: 1px solid var(--app-primary-line);
  font-size: 12.5px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.similar-chip:hover {
  background: var(--app-primary);
  color: #fff;
}
.thinking {
  color: var(--app-text-2);
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 2px 14px;
}
.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--app-primary);
}
.empty-conv {
  text-align: center;
  padding: 60px 20px;
}
.empty-icon {
  font-size: 34px;
  color: var(--app-primary);
  margin-bottom: 10px;
}
.empty-conv h3 {
  margin: 0 0 6px;
  font-size: 16px;
  font-weight: 500;
}
.hot-inline {
  margin-top: 20px;
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
  justify-content: center;
}
.input-bar {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  padding: 12px 16px 16px;
  border-top: 1px solid var(--app-border);
}

/* ---------- 右侧热问榜 ---------- */
.side-panel {
  width: 252px;
  flex: 0 0 252px;
  background: var(--app-card);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  padding: 16px;
  box-shadow: var(--app-shadow);
  overflow-y: auto;
}
.side-title {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}
.side-hint {
  margin: 5px 0 14px;
}
.hot-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.hot-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 6px;
  border-radius: 8px;
  cursor: pointer;
}
.hot-item:hover {
  background: var(--app-border-soft);
}
.hot-rank {
  width: 19px;
  height: 19px;
  flex: 0 0 auto;
  border-radius: 6px;
  background: var(--app-border-soft);
  color: var(--app-text-3);
  font-size: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.hot-rank.top {
  background: var(--app-primary);
  color: #fff;
}
.hot-q {
  flex: 1;
  font-size: 12.5px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.hot-count {
  font-size: 11.5px;
  color: var(--app-text-3);
}

@media (max-width: 1180px) {
  .side-panel {
    display: none;
  }
}

/* Markdown 渲染结果：知识库片段表格/标题密集，要显示成正常排版 */
.md-body :deep(p) {
  margin: 4px 0;
  line-height: 1.7;
}
.md-body :deep(h4),
.md-body :deep(h5),
.md-body :deep(h6) {
  margin: 8px 0 4px;
  font-size: 13px;
  font-weight: 500;
  line-height: 1.5;
}
.md-body :deep(ul),
.md-body :deep(ol) {
  margin: 4px 0;
  padding-left: 20px;
}
.md-body :deep(li) {
  margin: 2px 0;
  line-height: 1.7;
}
.md-body :deep(strong) {
  font-weight: 500;
  color: var(--app-text);
}
.md-body :deep(code) {
  background: rgba(0, 0, 0, 0.05);
  padding: 1px 4px;
  border-radius: 4px;
  font-size: 12px;
}
.md-body :deep(blockquote) {
  margin: 5px 0;
  padding: 3px 10px;
  border-left: 3px solid var(--app-primary-line);
  color: var(--app-text-2);
}
.md-body :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin: 7px 0 3px;
  font-size: 12px;
}
.md-body :deep(th),
.md-body :deep(td) {
  border: 1px solid var(--app-border);
  padding: 4px 7px;
  text-align: left;
}
.md-body :deep(th) {
  background: #f8fbfa;
  font-weight: 500;
}
.reason-group {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.reason-group :deep(.el-radio) {
  margin-right: 0;
}
</style>
