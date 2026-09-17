/**
 * 轻量 Markdown → HTML 渲染（零依赖）
 *
 * 为什么需要它：知识库片段与助手回答在界面上必须显示成「正常排版」，
 * 而不是 `**加粗**`、`| 尺码 | 胸围 |` 这样的源码。Markdown 只是索引层的中间表示，
 * 用户不该看到它。
 *
 * 安全：先 escapeHtml 再做标签替换，因此原文中的 `<script>` 之类会被转义成文本，
 * 天然免疫 XSS，不需要额外的 sanitize 依赖。
 *
 * 支持的语法（覆盖知识库文档与模型输出的常见写法）：
 * 标题 #~######、表格、无序/有序列表、加粗、斜体、行内代码、引用块、段落
 */

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** 行内渲染：先转义，再替换成标签（顺序不能反，否则标签会被一起转义） */
function inline(src) {
  return escapeHtml(src)
    .replace(/\\([\\`*_{}[\]()#+\-.!|])/g, '$1')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>')
}

/** 是否 Markdown 表格行：以 | 开头且至少 2 个竖线 */
function isTableRow(line) {
  const s = line.trim()
  return s.startsWith('|') && (s.match(/\|/g) || []).length >= 2
}

/** 是否表格的 `|---|---|` 分隔行 */
function isSepRow(line) {
  const s = line.trim()
  return /^[\s|:-]+$/.test(s) && s.includes('-')
}

/** 拆单元格：保护转义的 \| ，避免被当成列分隔符 */
function splitCells(line) {
  let s = line.trim()
  if (s.startsWith('|')) s = s.slice(1)
  if (s.endsWith('|')) s = s.slice(0, -1)
  return s
    .replace(/\\\|/g, '\u0000')
    .split('|')
    .map((c) => c.replace(/\u0000/g, '|').trim())
}

function renderTable(block) {
  const rows = []
  let sepSkipped = false
  for (const line of block) {
    if (!sepSkipped && isSepRow(line)) {
      sepSkipped = true
      continue
    }
    rows.push(splitCells(line))
  }
  if (!rows.length) return ''
  const head = rows.shift()
  const th = head.map((c) => `<th>${inline(c)}</th>`).join('')
  const body = rows
    .map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`)
    .join('')
  return `<table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table>`
}

/**
 * 把 Markdown 文本渲染成 HTML 字符串（供 v-html 使用）。
 */
export function renderRich(text) {
  if (text === null || text === undefined) return ''
  const lines = String(text).replace(/\r\n?/g, '\n').split('\n')
  const html = []
  let para = []
  let list = null

  const flushPara = () => {
    if (para.length) {
      html.push(`<p>${inline(para.join(' '))}</p>`)
      para = []
    }
  }
  const flushList = () => {
    if (list) {
      const items = list.items.map((it) => `<li>${inline(it)}</li>`).join('')
      html.push(`<${list.tag}>${items}</${list.tag}>`)
      list = null
    }
  }
  const flushAll = () => {
    flushPara()
    flushList()
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()

    if (!line) {
      flushAll()
      continue
    }

    // 表格：吞掉后续所有连续的表格行
    if (isTableRow(lines[i])) {
      const block = []
      while (i < lines.length && isTableRow(lines[i])) {
        block.push(lines[i])
        i++
      }
      i--
      flushAll()
      html.push(renderTable(block))
      continue
    }

    // 标题：卡片内从 h4 起，避免标题比界面层级还大
    const heading = /^(#{1,6})\s+(.*)$/.exec(line)
    if (heading) {
      flushAll()
      const level = Math.min(heading[1].length + 3, 6)
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`)
      continue
    }

    // 引用块
    if (/^>\s?/.test(line)) {
      flushAll()
      html.push(`<blockquote>${inline(line.replace(/^>\s?/, ''))}</blockquote>`)
      continue
    }

    // 无序 / 有序列表
    const asUl = /^[-*•]\s+(.*)$/.exec(line)
    const asOl = /^\d+[.)]\s+(.*)$/.exec(line)
    if (asUl || asOl) {
      flushPara()
      const tag = asUl ? 'ul' : 'ol'
      if (!list || list.tag !== tag) {
        flushList()
        list = { tag, items: [] }
      }
      list.items.push(asUl ? asUl[1] : asOl[1])
      continue
    }

    // 普通行：连续行并成一段（知识文档里同一段常被折成多行）
    flushList()
    para.push(line)
  }

  flushAll()
  return html.join('')
}

export default renderRich
