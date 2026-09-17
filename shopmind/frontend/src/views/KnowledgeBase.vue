<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h2 class="page-title">知识库管理</h2>
        <p class="page-sub">
          支持 Word / PDF / Markdown / 表格，以及图片型图册（自动 OCR 识别文字）
        </p>
      </div>
      <el-upload
        :auto-upload="false"
        :show-file-list="false"
        :on-change="onFilePicked"
        :accept="ACCEPT"
      >
        <el-button type="primary" :icon="Upload" :loading="uploading">
          {{ uploading ? '上传中…' : '上传文档' }}
        </el-button>
      </el-upload>
    </div>

    <!-- 概览指标 -->
    <div class="metric-grid" style="margin-bottom: 16px">
      <div class="metric">
        <div class="metric-label"><el-icon><Files /></el-icon>文档</div>
        <div class="metric-value">{{ stats?.total_documents ?? 0 }}<small>篇</small></div>
        <div class="metric-foot">已就绪 {{ stats?.ready_documents ?? 0 }} 篇</div>
      </div>
      <div class="metric">
        <div class="metric-label"><el-icon><Grid /></el-icon>向量片段</div>
        <div class="metric-value">{{ stats?.total_chunks ?? 0 }}<small>条</small></div>
        <div class="metric-foot">已写入 Chroma 向量库</div>
      </div>
      <div class="metric">
        <div class="metric-label"><el-icon><ChatDotRound /></el-icon>问答量</div>
        <div class="metric-value">{{ qStats?.total_queries ?? 0 }}<small>次</small></div>
        <div class="metric-foot">本次运行累计</div>
      </div>
      <div class="metric">
        <div class="metric-label"><el-icon><Timer /></el-icon>平均时延</div>
        <div class="metric-value">{{ qStats?.avg_latency_ms ?? 0 }}<small>ms</small></div>
        <div class="metric-foot">
          命中率 {{ ((qStats?.cache_hit_rate ?? 0) * 100).toFixed(0) }}%
        </div>
      </div>
    </div>

    <div class="app-card">
      <h3 class="card-title">文档列表</h3>

      <el-alert
        v-if="hasFailed"
        type="error"
        :closable="false"
        style="margin-bottom: 14px"
        title="有文档处理失败，请看「状态」与「错误信息」列，修正后点「重索引」"
      />

      <el-table :data="docs" v-loading="loading" empty-text="暂无文档，先上传一些商品资料吧">
        <el-table-column prop="filename" label="文件" min-width="220" show-overflow-tooltip>
          <template #default="{ row }">
            <div class="doc-name">
              <span>{{ row.filename }}</span>
              <span v-if="isOcr(row)" class="soft-tag">OCR</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="file_type" label="类型" width="82" />
        <el-table-column label="状态" width="104">
          <template #default="{ row }">
            <span class="soft-tag" :class="tagClass(row.status)">{{ statusText(row.status) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="chunk_count" label="片段" width="76" />
        <el-table-column prop="char_count" label="字符" width="86" />
        <el-table-column prop="error" label="错误信息" min-width="150" show-overflow-tooltip>
          <template #default="{ row }">
            <span class="muted">{{ row.error || '-' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="232" fixed="right">
          <template #default="{ row }">
            <el-button size="small" @click="onDownload(row)">下载原件</el-button>
            <el-button size="small" :loading="row._busy" @click="onReindex(row)">
              重索引
            </el-button>
            <el-popconfirm title="确认删除该文档及其向量与分块？" @confirm="onDelete(row)">
              <template #reference>
                <el-button size="small" type="danger" plain>删除</el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>

      <p class="muted" style="margin: 14px 0 0">
        说明：上传的原始文件按原格式原样保存，下载拿到的仍是原件（Word 还是 Word）；
        Markdown 只作为内部的索引中间表示，不会出现在界面上。
        改了切块策略后，旧文档需要点「重索引」才会应用新的切块方式。
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Upload, Files, Grid, ChatDotRound, Timer } from '@element-plus/icons-vue'
import {
  listDocuments,
  docStats,
  uploadDocument,
  reindexDocument,
  deleteDocument,
  downloadDocument,
  queryStats,
} from '@/api'

// 图片类也收：后端会走 OCR 从图册里抠文字
const ACCEPT = '.txt,.md,.pdf,.docx,.csv,.xlsx,.json,.png,.jpg,.jpeg,.bmp,.webp,.tif,.tiff'

const docs = ref([])
const stats = ref(null)
const qStats = ref(null)
const loading = ref(false)
const uploading = ref(false)

const hasFailed = computed(() => docs.value.some((d) => d.status === 'failed'))

function isOcr(row) {
  try {
    return !!JSON.parse(row.meta_json || '{}').used_ocr
  } catch {
    return false
  }
}

async function refresh(silent = false) {
  if (!silent) loading.value = true
  try {
    const [d, s, q] = await Promise.all([listDocuments(), docStats(), queryStats()])
    docs.value = d
    stats.value = s
    qStats.value = q
  } catch (e) {
    if (!silent) ElMessage.error(e.response?.data?.detail || '加载失败')
  } finally {
    if (!silent) loading.value = false
  }
}

// 上传/重索引后轮询，直到没有「处理中」的文档（避免用户误以为卡住）
let pollTimer = null
function pollStatus(maxSeconds = 180) {
  clearInterval(pollTimer)
  const started = Date.now()
  pollTimer = setInterval(async () => {
    await refresh(true)
    const pending = docs.value.some((d) => d.status === 'processing')
    const timeout = Date.now() - started > maxSeconds * 1000
    if (pending && !timeout) return
    clearInterval(pollTimer)
    if (pending) {
      ElMessage.warning('处理时间较长（OCR 或大文档较慢），可稍后刷新或点「重索引」')
      return
    }
    const failed = docs.value.filter((d) => d.status === 'failed')
    if (failed.length) ElMessage.error(`有 ${failed.length} 篇处理失败，请看「错误信息」列`)
    else ElMessage.success('处理完成')
  }, 2500)
}

onUnmounted(() => clearInterval(pollTimer))

function tagClass(s) {
  return s === 'ready' ? '' : s === 'processing' ? 'gray' : 'danger'
}
function statusText(s) {
  return { ready: '已就绪', processing: '处理中', failed: '失败' }[s] || s
}

async function onFilePicked(file) {
  uploading.value = true
  try {
    await uploadDocument(file.raw)
    ElMessage.success('已上传，正在后台解析与向量化…')
    await refresh()
    pollStatus()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '上传失败')
  } finally {
    uploading.value = false
  }
}

async function onReindex(row) {
  row._busy = true
  try {
    await reindexDocument(row.id)
    ElMessage.success('已重新索引，处理中…')
    await refresh()
    pollStatus()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '重索引失败')
  } finally {
    row._busy = false
  }
}

async function onDelete(row) {
  try {
    await deleteDocument(row.id)
    ElMessage.success('已删除')
    await refresh()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '删除失败')
  }
}

// 下载原件：后端返回原始文件（如 .docx），不是索引用的 Markdown
async function onDownload(row) {
  try {
    const res = await downloadDocument(row.id)
    const url = URL.createObjectURL(new Blob([res.data]))
    const a = document.createElement('a')
    a.href = url
    a.download = row.filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  } catch (e) {
    ElMessage.error('下载失败，请确认原始文件是否存在')
  }
}

onMounted(refresh)
</script>

<style scoped>
.doc-name {
  display: flex;
  align-items: center;
  gap: 8px;
}
.soft-tag.danger {
  background: #fdeeec;
  color: #b5473a;
  border-color: #f5d2cc;
}
</style>
