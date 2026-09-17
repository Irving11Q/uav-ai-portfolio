<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h2 class="page-title">系统设置</h2>
        <p class="page-sub">
          调整大模型与检索参数，<strong>保存后立即生效</strong>，无需重启服务
        </p>
      </div>
      <div class="head-actions">
        <el-button :icon="Refresh" @click="load">重新读取</el-button>
        <el-button type="primary" :loading="saving" @click="save">保存修改</el-button>
      </div>
    </div>

    <el-alert
      v-if="changedCount"
      type="warning"
      :closable="false"
      style="margin-bottom: 16px"
      :title="`有 ${changedCount} 项待保存的修改`"
      description="点击「保存修改」后写入数据库并立即对新的提问生效。"
    />

    <div v-for="g in groups" :key="g.name" class="app-card">
      <h3 class="card-title">{{ g.name }}</h3>
      <div class="setting-list">
        <div v-for="item in g.items" :key="item.key" class="setting-row">
          <div class="setting-info">
            <div class="setting-label">
              {{ item.label }}
              <span v-if="isChanged(item)" class="soft-tag">已修改</span>
              <span v-else-if="item.overridden" class="soft-tag gray">后台已覆盖</span>
            </div>
            <div class="setting-desc">{{ item.desc }}</div>
            <div class="setting-key">
              {{ item.key }} · 默认 {{ item.default }}
            </div>
          </div>
          <div class="setting-control">
            <el-switch
              v-if="item.type === 'bool'"
              v-model="draft[item.key]"
              :disabled="!isChanged(item) && false"
            />
            <el-select
              v-else-if="item.type === 'enum'"
              v-model="draft[item.key]"
              filterable
              allow-create
              default-first-option
              style="width: 210px"
            >
              <el-option v-for="o in item.options || []" :key="o" :label="o" :value="o" />
            </el-select>
            <el-input-number
              v-else
              v-model="draft[item.key]"
              :min="item.min"
              :max="item.max"
              :step="item.step || 1"
              :precision="item.type === 'float' ? 2 : 0"
              controls-position="right"
              style="width: 160px"
            />
          </div>
        </div>
      </div>
    </div>

    <div class="app-card">
      <h3 class="card-title">恢复默认</h3>
      <p class="setting-desc" style="margin-top: 0">
        清空后台的所有覆盖，全部回到 <code>.env</code> 里的默认值。
        用于「参数调乱了，一键回到已知可用状态」。
      </p>
      <el-popconfirm
        title="确定要清空全部后台配置，恢复为 .env 默认值吗？"
        confirm-button-text="确定恢复"
        cancel-button-text="取消"
        @confirm="resetAll"
      >
        <template #reference>
          <el-button type="danger" plain>恢复默认值</el-button>
        </template>
      </el-popconfirm>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { getSettings, updateSettings, resetSettings } from '@/api'

const items = ref([])
const original = ref({})
const draft = ref({})
const saving = ref(false)

// 按 group 分组渲染，组内顺序沿用后端 SPEC 的顺序
const groups = computed(() => {
  const map = new Map()
  for (const it of items.value) {
    const name = it.group || '其他'
    if (!map.has(name)) map.set(name, [])
    map.get(name).push(it)
  }
  return [...map.entries()].map(([name, list]) => ({ name, items: list }))
})

const changedCount = computed(
  () => items.value.filter((it) => isChanged(it)).length
)

function isChanged(item) {
  return draft.value[item.key] !== original.value[item.key]
}

function fill(data) {
  items.value = data.items || []
  original.value = Object.fromEntries(items.value.map((i) => [i.key, i.value]))
  draft.value = { ...original.value }
}

async function load() {
  try {
    fill(await getSettings())
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '读取配置失败')
  }
}

async function save() {
  const values = {}
  for (const it of items.value) {
    if (isChanged(it)) values[it.key] = draft.value[it.key]
  }
  if (!Object.keys(values).length) {
    ElMessage.info('没有需要保存的修改')
    return
  }
  saving.value = true
  try {
    fill(await updateSettings(values))
    ElMessage.success(`已保存 ${Object.keys(values).length} 项配置，新的提问即刻生效`)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '保存失败')
  } finally {
    saving.value = false
  }
}

async function resetAll() {
  try {
    fill(await resetSettings())
    ElMessage.success('已恢复为默认值')
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '恢复失败')
  }
}

onMounted(load)
</script>

<style scoped>
.head-actions {
  display: flex;
  gap: 10px;
}
.setting-list {
  display: flex;
  flex-direction: column;
}
.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 14px 0;
  border-bottom: 1px dashed var(--app-border-soft);
}
.setting-row:last-child {
  border-bottom: none;
  padding-bottom: 0;
}
.setting-info {
  min-width: 0;
}
.setting-label {
  font-size: 13.5px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.setting-desc {
  margin-top: 4px;
  font-size: 12.5px;
  color: var(--app-text-2);
  line-height: 1.6;
}
.setting-key {
  margin-top: 4px;
  font-size: 11.5px;
  color: var(--app-text-3);
  font-family: ui-monospace, Menlo, Consolas, monospace;
}
.setting-control {
  flex: 0 0 auto;
}
code {
  background: var(--app-border-soft);
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 12px;
}
</style>
