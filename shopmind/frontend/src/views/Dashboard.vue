<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h2 class="page-title">数据看板</h2>
        <p class="page-sub">
          知识库规模、使用情况与检索质量一屏总览
        </p>
      </div>
      <el-button :icon="Refresh" :loading="loading" @click="loadAll">刷新</el-button>
    </div>

    <!-- 核心指标 -->
    <div class="metric-grid">
      <div v-for="m in metrics" :key="m.label" class="metric">
        <div class="metric-label">
          <el-icon><component :is="m.icon" /></el-icon>{{ m.label }}
        </div>
        <div class="metric-value">
          {{ m.value }}<small v-if="m.unit">{{ m.unit }}</small>
        </div>
        <div class="metric-foot">{{ m.foot }}</div>
      </div>
    </div>

    <!-- 趋势 + 热问榜 -->
    <div class="two-col">
      <div class="app-card">
        <h3 class="card-title">近 7 天提问量</h3>
        <div v-if="trend.length" class="trend">
          <div v-for="t in trend" :key="t.date" class="trend-col">
            <div class="trend-num">{{ t.count }}</div>
            <div class="trend-track">
              <div class="trend-bar" :style="{ height: barHeight(t.count) }"></div>
            </div>
            <div class="trend-day">{{ t.date.slice(5) }}</div>
          </div>
        </div>
        <el-empty v-else description="暂无提问数据" :image-size="70" />
      </div>

      <div class="app-card">
        <h3 class="card-title">热问榜 Top 10</h3>
        <div v-if="hot.length" class="hot-list">
          <div v-for="(h, i) in hot" :key="i" class="hot-row">
            <span class="hot-rank" :class="{ top: i < 3 }">{{ i + 1 }}</span>
            <span class="hot-q" :title="h.question">{{ h.question }}</span>
            <span class="hot-count">{{ h.count }} 次</span>
            <div class="hot-bar-wrap">
              <div class="hot-bar" :style="{ width: hotWidth(h.ratio) }"></div>
            </div>
          </div>
        </div>
        <el-empty v-else description="还没有提问记录" :image-size="70" />
      </div>
    </div>

    <!-- 文档明细 -->
    <div class="app-card">
      <h3 class="card-title">文档明细</h3>
      <el-table :data="documents" style="width: 100%">
        <el-table-column prop="filename" label="文档" min-width="240" show-overflow-tooltip />
        <el-table-column prop="status" label="状态" width="110">
          <template #default="{ row }">
            <span class="soft-tag" :class="{ gray: row.status !== 'ready' }">
              {{ statusText(row.status) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="chunk_count" label="分块数" width="100" />
        <el-table-column prop="char_count" label="字符数" width="110" />
      </el-table>
    </div>

    <p class="muted" style="margin-top: 14px">
      注：带「本次运行」标记的指标来自进程内埋点，服务重启后会重新累计；
      文档、分块、用户、提问量、反馈数均为数据库口径，永久保留。
    </p>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Refresh,
  Document,
  Grid,
  User,
  ChatDotRound,
  Cpu,
  Timer,
  Star,
} from '@element-plus/icons-vue'
import {
  dashboard,
  dashboardTrend,
  dashboardHotQuestions,
  dashboardDocuments,
} from '@/api'

const loading = ref(false)
const stats = ref({})
const trend = ref([])
const hot = ref([])
const documents = ref([])

const metrics = computed(() => {
  const s = stats.value
  const pct = (v) => `${((v || 0) * 100).toFixed(1)}`
  return [
    {
      label: '文档数',
      icon: Document,
      value: s.documents ?? 0,
      unit: '篇',
      foot: `就绪 ${s.ready_documents ?? 0} 篇${s.failed_documents ? ` / 失败 ${s.failed_documents} 篇` : ''}`,
    },
    {
      label: '分块数',
      icon: Grid,
      value: s.chunks ?? 0,
      unit: '块',
      foot: `约 ${((s.char_count || 0) / 1000).toFixed(1)} 千字已入库`,
    },
    {
      label: '注册用户',
      icon: User,
      value: s.users ?? 0,
      unit: '人',
      foot: `会话 ${s.sessions ?? 0} 个`,
    },
    {
      label: '问答量',
      icon: ChatDotRound,
      value: s.questions ?? 0,
      unit: '次',
      foot: `本次运行 ${s.total_queries ?? 0} 次`,
    },
    {
      label: '缓存命中率',
      icon: Cpu,
      value: pct(s.cache_hit_rate),
      unit: '%',
      foot: `本次运行命中 ${s.cache_hits ?? 0} 次`,
    },
    {
      label: '平均响应',
      icon: Timer,
      value: Math.round(s.avg_latency_ms ?? 0),
      unit: 'ms',
      foot: `检索 ${Math.round(s.avg_retrieval_ms ?? 0)}ms / 生成 ${Math.round(s.avg_llm_ms ?? 0)}ms`,
    },
    {
      label: '满意度',
      icon: Star,
      value: pct(s.satisfaction_rate),
      unit: '%',
      foot: `赞 ${s.feedback_up ?? 0} / 踩 ${s.feedback_down ?? 0}`,
    },
  ]
})

const maxTrend = computed(() => Math.max(1, ...trend.value.map((t) => t.count)))
const maxRatio = computed(() => Math.max(0.0001, ...hot.value.map((h) => h.ratio)))

function barHeight(count) {
  return `${Math.max(3, (count / maxTrend.value) * 100)}%`
}
function hotWidth(ratio) {
  return `${Math.max(4, (ratio / maxRatio.value) * 100)}%`
}
function statusText(s) {
  return { ready: '已就绪', processing: '处理中', failed: '失败' }[s] || s
}

async function loadAll() {
  loading.value = true
  try {
    const [s, t, h, d] = await Promise.all([
      dashboard(),
      dashboardTrend(7),
      dashboardHotQuestions(10, 30),
      dashboardDocuments(),
    ])
    stats.value = s
    trend.value = t
    hot.value = h
    documents.value = d
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '加载看板数据失败')
  } finally {
    loading.value = false
  }
}

onMounted(loadAll)
</script>

<style scoped>
.two-col {
  display: grid;
  /* minmax(0, 1fr)：1fr 的最小值是 auto，宽内容会把列撑破、两列不等宽 */
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 16px;
  margin: 16px 0;
}
/* 全局 `.app-card + .app-card { margin-top: 16px }` 是给纵向堆叠准备的，
   在网格里会让第二张卡片下沉 16px（实测 395.2 vs 411.2）。同行间距交给 gap。 */
.two-col > .app-card + .app-card {
  margin-top: 0;
}
@media (max-width: 1000px) {
  .two-col {
    grid-template-columns: minmax(0, 1fr);
  }
  /* 单列时恢复纵向间距 */
  .two-col > .app-card + .app-card {
    margin-top: 16px;
  }
}

/* 趋势柱状图（纯 CSS，不引图表库） */
.trend {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  height: 180px;
}
.trend-col {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
}
.trend-num {
  font-size: 12px;
  color: var(--app-text-2);
  margin-bottom: 4px;
}
.trend-track {
  flex: 1;
  width: 100%;
  display: flex;
  align-items: flex-end;
}
.trend-bar {
  width: 100%;
  background: var(--app-primary);
  opacity: 0.85;
  border-radius: 6px 6px 3px 3px;
  transition: height 0.4s ease;
}
.trend-day {
  margin-top: 8px;
  font-size: 11.5px;
  color: var(--app-text-3);
}

/* 热问榜 */
.hot-list {
  display: flex;
  flex-direction: column;
  gap: 9px;
}
.hot-row {
  display: grid;
  grid-template-columns: 22px 1fr 58px 84px;
  align-items: center;
  gap: 9px;
}
.hot-rank {
  width: 20px;
  height: 20px;
  border-radius: 6px;
  background: var(--app-border-soft);
  color: var(--app-text-3);
  font-size: 11.5px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.hot-rank.top {
  background: var(--app-primary);
  color: #fff;
}
.hot-q {
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.hot-count {
  font-size: 12px;
  color: var(--app-text-2);
  text-align: right;
}
.hot-bar-wrap {
  height: 6px;
  background: var(--app-border-soft);
  border-radius: 3px;
  overflow: hidden;
}
.hot-bar {
  height: 100%;
  background: var(--app-primary);
  opacity: 0.75;
  border-radius: 3px;
}
</style>
