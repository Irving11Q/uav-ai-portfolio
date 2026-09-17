<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h2 class="page-title">审计日志</h2>
        <p class="page-sub">谁在何时做了什么 —— 含每次提问的原文，可按人、动作、关键词与时间检索</p>
      </div>
      <el-button :icon="Download" :loading="exporting" @click="exportCsv">
        导出 CSV
      </el-button>
    </div>

    <div class="app-card">
      <div class="filters">
        <el-select
          v-model="filters.username"
          placeholder="全部用户"
          clearable
          filterable
          style="width: 160px"
        >
          <el-option
            v-for="u in userOptions"
            :key="u.username"
            :label="`${u.username}（${u.count}）`"
            :value="u.username"
          />
        </el-select>

        <el-select
          v-model="filters.action"
          placeholder="全部动作"
          clearable
          style="width: 160px"
        >
          <el-option
            v-for="a in actionOptions"
            :key="a.action"
            :label="`${a.label}（${a.count}）`"
            :value="a.action"
          />
        </el-select>

        <el-select v-model="filters.days" placeholder="时间范围" clearable style="width: 140px">
          <el-option label="最近 1 天" :value="1" />
          <el-option label="最近 7 天" :value="7" />
          <el-option label="最近 30 天" :value="30" />
        </el-select>

        <el-input
          v-model="filters.keyword"
          placeholder="搜索问题内容 / 描述 / 对象"
          clearable
          style="width: 260px"
          @keyup.enter="reload"
        >
          <template #prefix><el-icon><Search /></el-icon></template>
        </el-input>

        <el-button type="primary" @click="reload">查询</el-button>
        <el-button @click="resetFilters">重置</el-button>
      </div>

      <el-table :data="rows" v-loading="loading" style="width: 100%">
        <el-table-column prop="created_at" label="时间" width="160">
          <template #default="{ row }">{{ fmt(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="用户" width="130">
          <template #default="{ row }">
            <span v-if="row.username">{{ row.username }}</span>
            <span v-else class="muted">未登录</span>
          </template>
        </el-table-column>
        <el-table-column prop="action_label" label="动作" width="110">
          <template #default="{ row }">
            <span class="soft-tag" :class="{ gray: row.action === 'login_failed' }">
              {{ row.action_label }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="问题 / 描述" min-width="300">
          <template #default="{ row }">
            <div v-if="row.question" class="q-cell">{{ row.question }}</div>
            <div class="muted">{{ row.detail }}</div>
          </template>
        </el-table-column>
        <el-table-column prop="ip" label="IP" width="130">
          <template #default="{ row }">
            <span class="muted">{{ row.ip || '-' }}</span>
          </template>
        </el-table-column>
      </el-table>

      <div class="pager">
        <span class="muted">共 {{ total }} 条</span>
        <el-pagination
          v-model:current-page="page"
          v-model:page-size="pageSize"
          :page-sizes="[20, 50, 100]"
          :total="total"
          layout="sizes, prev, pager, next"
          @current-change="load"
          @size-change="reload"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Download, Search } from '@element-plus/icons-vue'
import { auditLogs, auditActions, auditUsers, exportAuditLogs } from '@/api'

const loading = ref(false)
const exporting = ref(false)
const rows = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const actionOptions = ref([])
const userOptions = ref([])

const filters = reactive({ username: '', action: '', days: null, keyword: '' })

function params() {
  const p = { page: page.value, page_size: pageSize.value }
  if (filters.username) p.username = filters.username
  if (filters.action) p.action = filters.action
  if (filters.days) p.days = filters.days
  if (filters.keyword) p.keyword = filters.keyword
  return p
}

async function load() {
  loading.value = true
  try {
    const d = await auditLogs(params())
    rows.value = d.items || []
    total.value = d.total || 0
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '加载审计日志失败')
  } finally {
    loading.value = false
  }
}

function reload() {
  page.value = 1
  load()
}

function resetFilters() {
  filters.username = ''
  filters.action = ''
  filters.days = null
  filters.keyword = ''
  reload()
}

async function exportCsv() {
  exporting.value = true
  try {
    const p = params()
    delete p.page
    delete p.page_size
    const res = await exportAuditLogs(p)
    const blob = new Blob([res.data], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `审计日志_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
    ElMessage.success('已导出 CSV（UTF-8 BOM，Excel 打开不乱码）')
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '导出失败')
  } finally {
    exporting.value = false
  }
}

function fmt(t) {
  if (!t) return '-'
  return t.replace('T', ' ').slice(0, 19)
}

onMounted(async () => {
  try {
    const [a, u] = await Promise.all([auditActions(), auditUsers()])
    actionOptions.value = a
    userOptions.value = u
  } catch {
    /* 下拉选项拿不到也不影响主表 */
  }
  load()
})
</script>

<style scoped>
.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 16px;
}
.q-cell {
  font-size: 13px;
  margin-bottom: 2px;
  word-break: break-word;
}
.pager {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 14px;
}
</style>
