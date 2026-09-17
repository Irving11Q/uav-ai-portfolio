<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h2 class="page-title">反馈与调优</h2>
        <p class="page-sub">把「踩」的回答沉淀成 bad case，逐条排查后标记处理，形成调优闭环</p>
      </div>
      <el-button :icon="Refresh" :loading="loading" @click="loadAll">刷新</el-button>
    </div>

    <div class="metric-grid">
      <div class="metric">
        <div class="metric-label"><el-icon><CircleCheck /></el-icon>满意（赞）</div>
        <div class="metric-value">{{ summary.up ?? 0 }}<small>条</small></div>
        <div class="metric-foot">来自真实用户点击</div>
      </div>
      <div class="metric">
        <div class="metric-label"><el-icon><CircleClose /></el-icon>不满意（踩）</div>
        <div class="metric-value">{{ summary.down ?? 0 }}<small>条</small></div>
        <div class="metric-foot">待处理 {{ unresolvedCount }} 条</div>
      </div>
      <div class="metric">
        <div class="metric-label"><el-icon><Star /></el-icon>满意度</div>
        <div class="metric-value">
          {{ ((summary.satisfaction_rate || 0) * 100).toFixed(1) }}<small>%</small>
        </div>
        <div class="metric-foot">赞 /（赞 + 踩）</div>
      </div>
    </div>

    <div v-if="summary.reason_breakdown?.length" class="app-card" style="margin-top: 16px">
      <h3 class="card-title">「踩」的原因分布</h3>
      <div class="reason-list">
        <div v-for="r in summary.reason_breakdown" :key="r.reason" class="reason-row">
          <span class="reason-name">{{ r.reason }}</span>
          <div class="reason-track">
            <div class="reason-bar" :style="{ width: reasonWidth(r.count) }"></div>
          </div>
          <span class="reason-count">{{ r.count }}</span>
        </div>
      </div>
    </div>

    <div class="app-card">
      <h3 class="card-title">反馈明细</h3>
      <div class="filters">
        <el-radio-group v-model="filters.rating" @change="reload">
          <el-radio-button label="down">只看不满意</el-radio-button>
          <el-radio-button label="up">只看满意</el-radio-button>
          <el-radio-button label="">全部</el-radio-button>
        </el-radio-group>
        <el-switch
          v-model="filters.onlyUnresolved"
          active-text="只看未处理"
          @change="reload"
        />
        <el-input
          v-model="filters.keyword"
          placeholder="搜索问题 / 答案 / 备注"
          clearable
          style="width: 250px"
          @keyup.enter="reload"
        >
          <template #prefix><el-icon><Search /></el-icon></template>
        </el-input>
        <el-button type="primary" @click="reload">查询</el-button>
      </div>

      <div v-if="items.length" class="fb-list">
        <div v-for="f in items" :key="f.id" class="fb-item">
          <div class="fb-head">
            <span class="soft-tag" :class="{ gray: f.rating === 'up' }">
              {{ f.rating === 'up' ? '满意' : '不满意' }}
            </span>
            <span v-if="f.reason" class="soft-tag gray">{{ f.reason }}</span>
            <span class="muted">{{ f.username || '匿名' }} · {{ fmt(f.created_at) }}</span>
            <span class="fb-head-right">
              <span v-if="f.resolved" class="soft-tag">已处理</span>
              <el-button
                size="small"
                :type="f.resolved ? 'info' : 'primary'"
                plain
                @click="openResolve(f)"
              >
                {{ f.resolved ? '取消标记' : '标记已处理' }}
              </el-button>
            </span>
          </div>

          <div class="fb-q">
            <span class="fb-tag">问</span>{{ f.question || '（未记录问题）' }}
          </div>
          <div class="fb-a">
            <span class="fb-tag">答</span>
            <span :class="{ clamp: !expanded.has(f.id) }">{{ f.answer }}</span>
            <el-button
              v-if="(f.answer || '').length > 120"
              link
              type="primary"
              size="small"
              @click="toggle(f.id)"
            >
              {{ expanded.has(f.id) ? '收起' : '展开' }}
            </el-button>
          </div>

          <div v-if="f.comment" class="fb-comment">用户备注：{{ f.comment }}</div>
          <div v-if="f.resolved_note" class="fb-note">处理备注：{{ f.resolved_note }}</div>
        </div>
      </div>
      <el-empty v-else description="暂无符合条件的反馈" :image-size="80" />

      <div class="pager">
        <span class="muted">共 {{ total }} 条</span>
        <el-pagination
          v-model:current-page="page"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50]"
          :total="total"
          layout="sizes, prev, pager, next"
          @current-change="loadList"
          @size-change="reload"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh, Search, Star, CircleCheck, CircleClose } from '@element-plus/icons-vue'
import { feedbackSummary, listBadCases, resolveBadCase } from '@/api'

const loading = ref(false)
const summary = ref({})
const items = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(10)
const expanded = ref(new Set())

const filters = reactive({ rating: 'down', onlyUnresolved: false, keyword: '' })

const unresolvedCount = computed(() => summary.value.down ?? 0)
const maxReason = computed(() =>
  Math.max(1, ...(summary.value.reason_breakdown || []).map((r) => r.count))
)

function reasonWidth(count) {
  return `${Math.max(6, (count / maxReason.value) * 100)}%`
}

function fmt(t) {
  return t ? t.replace('T', ' ').slice(0, 19) : '-'
}

function toggle(id) {
  const s = new Set(expanded.value)
  if (s.has(id)) s.delete(id)
  else s.add(id)
  expanded.value = s
}

async function loadList() {
  loading.value = true
  try {
    const params = { page: page.value, page_size: pageSize.value }
    if (filters.rating) params.rating = filters.rating
    if (filters.onlyUnresolved) params.only_unresolved = true
    if (filters.keyword) params.keyword = filters.keyword
    const d = await listBadCases(params)
    items.value = d.items || []
    total.value = d.total || 0
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '加载反馈失败')
  } finally {
    loading.value = false
  }
}

function reload() {
  page.value = 1
  loadList()
}

async function loadAll() {
  try {
    summary.value = await feedbackSummary()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '加载汇总失败')
  }
  await loadList()
}

async function openResolve(f) {
  if (f.resolved) {
    await resolveBadCase(f.id, null, false)
    ElMessage.success('已取消处理标记')
    loadAll()
    return
  }
  try {
    const { value } = await ElMessageBox.prompt(
      '可以记一下这次怎么处理的（例如：补充了尺码表 / 调整了分块策略），方便以后回看。',
      '标记为已处理',
      {
        confirmButtonText: '确认',
        cancelButtonText: '取消',
        inputPlaceholder: '处理备注（可留空）',
        inputValue: f.resolved_note || '',
      }
    )
    await resolveBadCase(f.id, value, true)
    ElMessage.success('已标记为处理完成')
    loadAll()
  } catch {
    /* 用户取消 */
  }
}

onMounted(loadAll)
</script>

<style scoped>
.filters {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 14px;
  margin-bottom: 16px;
}
.fb-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.fb-item {
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  padding: 14px 16px;
  background: #fdfefe;
}
.fb-head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}
.fb-head-right {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}
.fb-tag {
  display: inline-block;
  width: 20px;
  height: 20px;
  line-height: 20px;
  text-align: center;
  border-radius: 6px;
  background: var(--app-primary-soft);
  color: var(--app-primary-dark);
  font-size: 11.5px;
  margin-right: 7px;
  flex: 0 0 auto;
}
.fb-q {
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 7px;
  display: flex;
  align-items: baseline;
  line-height: 1.65;
}
.fb-a {
  font-size: 13px;
  color: var(--app-text-2);
  line-height: 1.7;
}
.clamp {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.fb-comment,
.fb-note {
  margin-top: 8px;
  font-size: 12.5px;
  color: var(--app-text-2);
  background: var(--app-border-soft);
  border-radius: 8px;
  padding: 6px 10px;
}
.fb-note {
  background: var(--app-primary-soft);
  color: var(--app-primary-dark);
}
.reason-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.reason-row {
  display: grid;
  grid-template-columns: 170px 1fr 40px;
  align-items: center;
  gap: 12px;
}
.reason-name {
  font-size: 13px;
}
.reason-track {
  height: 8px;
  background: var(--app-border-soft);
  border-radius: 4px;
  overflow: hidden;
}
.reason-bar {
  height: 100%;
  background: var(--app-down);
  opacity: 0.75;
  border-radius: 4px;
}
.reason-count {
  font-size: 12.5px;
  color: var(--app-text-2);
  text-align: right;
}
.pager {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 14px;
}
</style>
