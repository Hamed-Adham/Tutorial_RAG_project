"use client"

import React, { useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Message, Citation } from "@/types"
import {
  User,
  Bot,
  Zap,
  BookOpen,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Copy,
  Check,
  ExternalLink,
  DollarSign,
  Reply,
  MessageSquare
} from "lucide-react"

interface ChatMessageProps {
  message: Message
  isStreaming?: boolean
  streamingStatus?: string | null
  onSelectAsContext?: (message: Message) => void
}

/**
 * 3-Dot Triangle Cluster Loader
 * Matches modern AI interfaces (Claude/ChatGPT/Antigravity) as requested by user.
 */
export function ThinkingDots({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <span className={`inline-flex items-center justify-center shrink-0 ${className}`}>
      <svg
        className="w-full h-full animate-spin [animation-duration:4s] text-current"
        viewBox="0 0 24 24"
        fill="currentColor"
      >
        <circle cx="12" cy="5" r="2.4" className="animate-pulse" />
        <circle cx="5" cy="18" r="2.4" className="animate-pulse [animation-delay:200ms]" />
        <circle cx="19" cy="18" r="2.4" className="animate-pulse [animation-delay:400ms]" />
      </svg>
    </span>
  )
}

export const FILENAME_TO_CHAPTER: Record<string, { ch: number; name: string }> = {
  "C706_olom_haftom-7-144-2-5.pdf": { ch: 1, name: "تفکر و تجربه" },
  "C706_olom_haftom-7-144-6-11.pdf": { ch: 2, name: "اندازه‌گیری در علوم و ابزارهای آن" },
  "C706_olom_haftom-7-144-13-24.pdf": { ch: 3, name: "اتم‌ها، الفبای مواد" },
  "C706_olom_haftom-7-144-25-34.pdf": { ch: 4, name: "مواد پیرامون ما" },
  "C706_olom_haftom-7-144-36-47.pdf": { ch: 5, name: "از معدن تا خانه" },
  "C706_olom_haftom-7-144-48-55.pdf": { ch: 6, name: "سفر آب روی زمین" },
  "C706_olom_haftom-7-144-56-64.pdf": { ch: 7, name: "سفر آب درون زمین" },
  "C706_olom_haftom-7-144-66-76.pdf": { ch: 8, name: "انرژی و تبدیل‌های آن" },
  "C706_olom_haftom-7-144-77-85.pdf": { ch: 9, name: "منابع انرژی" },
  "C706_olom_haftom-7-144-86-97.pdf": { ch: 10, name: "گرما و بهینه‌سازی مصرف انرژی" },
  "C706_olom_haftom-7-144-99-106.pdf": { ch: 11, name: "یاخته و سازمان‌بندی آن" },
  "C706_olom_haftom-7-144-107-115.pdf": { ch: 12, name: "سفرهٔ سلامت" },
  "C706_olom_haftom-7-144-116-123.pdf": { ch: 13, name: "سفر غذا" },
  "C706_olom_haftom-7-144-124-131.pdf": { ch: 14, name: "گردش مواد" },
  "C706_olom_haftom-7-144-132-137.pdf": { ch: 15, name: "تبادل با محیط" },
}

export const CHAPTER_TO_FILENAME: Record<number, string> = {
  1: "C706_olom_haftom-7-144-2-5.pdf",
  2: "C706_olom_haftom-7-144-6-11.pdf",
  3: "C706_olom_haftom-7-144-13-24.pdf",
  4: "C706_olom_haftom-7-144-25-34.pdf",
  5: "C706_olom_haftom-7-144-36-47.pdf",
  6: "C706_olom_haftom-7-144-48-55.pdf",
  7: "C706_olom_haftom-7-144-56-64.pdf",
  8: "C706_olom_haftom-7-144-66-76.pdf",
  9: "C706_olom_haftom-7-144-77-85.pdf",
  10: "C706_olom_haftom-7-144-86-97.pdf",
  11: "C706_olom_haftom-7-144-99-106.pdf",
  12: "C706_olom_haftom-7-144-107-115.pdf",
  13: "C706_olom_haftom-7-144-116-123.pdf",
  14: "C706_olom_haftom-7-144-124-131.pdf",
  15: "C706_olom_haftom-7-144-132-137.pdf",
}

export function formatChapterCitationLabel(filename: string, pages?: string | number): string {
  const clean = filename.trim().split("#")[0].replace(/^\//, "")
  const info = FILENAME_TO_CHAPTER[clean]
  if (info) {
    if (pages) {
      return `فصل ${info.ch}، صفحه ${pages}`
    }
    return `فصل ${info.ch} (${info.name})`
  }
  return pages ? `صفحه ${pages}` : "کتاب درسی"
}

export function resolvePdfUrl(rawUrl: string, fallbackText?: string): string {
  if (!rawUrl) return "#"
  // Remove all whitespace that breaks markdown links in CommonMark / GFM
  let cleanUrl = rawUrl.replace(/\s+/g, "")

  // Collapse repeated or malformed api prefixes
  cleanUrl = cleanUrl.replace(/(?:\/?api\/v1\/pdf\/)+/g, "/api/v1/pdf/")
  cleanUrl = cleanUrl.replace(/^\/?api\/v1\/pdf\//, "")

  // Extract page fragment if present (#page=X, ?page=X, or from fallback text)
  const pageMatch = cleanUrl.match(/[#&?]page=(\d+)/) || (fallbackText ? fallbackText.match(/(?:صفحات?|صفحه)\s*([0-9]+)/) : null)
  const page = pageMatch ? pageMatch[1] : ""
  const pageFragment = page ? `#page=${page}` : ""

  // Strip page fragment from cleanUrl for matching base file/chapter
  const targetBase = cleanUrl.split("#")[0].split("?")[0]

  // Direct match in FILENAME_TO_CHAPTER
  if (FILENAME_TO_CHAPTER[targetBase]) {
    return `/api/v1/pdf/${targetBase}${pageFragment}`
  }

  // Extract C706 filename
  const c706Match = targetBase.match(/(C706_olom_haftom[^\s#?)]+\.pdf)/)
  if (c706Match && FILENAME_TO_CHAPTER[c706Match[1]]) {
    return `/api/v1/pdf/${c706Match[1]}${pageFragment}`
  }

  // Extract chapter number from URL or fallback label (e.g. فصل 15 or فصل 3)
  const chMatch = targetBase.match(/فصل\s*(\d+)/) || (fallbackText ? fallbackText.match(/فصل\s*(\d+)/) : null)
  if (chMatch) {
    const chNum = parseInt(chMatch[1], 10)
    const matchedFile = CHAPTER_TO_FILENAME[chNum]
    if (matchedFile) {
      return `/api/v1/pdf/${matchedFile}${pageFragment}`
    }
  }

  if (rawUrl.startsWith("http://") || rawUrl.startsWith("https://")) {
    return rawUrl.trim()
  }

  return `/api/v1/pdf/${targetBase.replace(/^\//, "")}${pageFragment}`
}

/**
 * Strip and convert raw LaTeX syntax and messy formula artifacts into clean, readable
 * plain text, unicode symbols, and bold markdown formatting.
 */
function cleanLatexAndFormulas(raw: string): string {
  if (!raw) return ""
  let text = raw

  // 1. Convert LaTeX text/style wrappers: \text{...}, \mathrm{...}, \operatorname{...}, \mathbf{...}, \mathit{...}
  text = text.replace(/\\(?:text|mathrm|operatorname)\{([^}]+)\}/g, "$1")
  text = text.replace(/\\mathbf\{([^}]+)\}/g, "**$1**")
  text = text.replace(/\\mathit\{([^}]+)\}/g, "*$1*")

  // 2. Fractions: \frac{num}{den} -> (num / den) or num / den
  for (let i = 0; i < 3; i++) {
    text = text.replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "($1 / $2)")
  }
  // Simplify outer parentheses after '=' like '= (m / V)' -> '= m / V'
  text = text.replace(/=\s*\(([^\/()]+)\s*\/\s*([^\/()]+)\)/g, "= $1 / $2")

  // 3. Greek and scientific symbols
  const symbolMap: Record<string, string> = {
    "\\\\rho\\b": "ρ",
    "\\\\Delta\\b": "Δ",
    "\\\\delta\\b": "δ",
    "\\\\pi\\b": "π",
    "\\\\alpha\\b": "α",
    "\\\\beta\\b": "β",
    "\\\\gamma\\b": "γ",
    "\\\\theta\\b": "θ",
    "\\\\lambda\\b": "λ",
    "\\\\mu\\b": "μ",
    "\\\\sigma\\b": "σ",
    "\\\\omega\\b": "ω",
    "\\\\times\\b": "×",
    "\\\\cdot\\b": "·",
    "\\\\approx\\b": "≈",
    "\\\\pm\\b": "±",
    "\\\\leq?\\b": "≤",
    "\\\\geq?\\b": "≥",
    "\\\\neq?\\b": "≠",
    "\\\\infty\\b": "∞",
    "\\\\degree\\b": "°",
    "\\\\circ\\b": "°",
  }
  for (const [pattern, replacement] of Object.entries(symbolMap)) {
    text = text.replace(new RegExp(pattern, "g"), replacement)
  }

  // 4. Superscripts
  text = text.replace(/\^2\b|\^\{2\}/g, "²")
  text = text.replace(/\^3\b|\^\{3\}/g, "³")
  text = text.replace(/\^1\b|\^\{1\}/g, "¹")
  text = text.replace(/\^0\b|\^\{0\}/g, "⁰")

  // 5. Block math ($$...$$) -> bold clean formula
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
    const clean = math.replace(/\s+/g, " ").trim()
    return clean ? ` **${clean}** ` : ""
  })

  // 6. Inline math ($...$ or $...\$ or \$...\$) -> bold variable/symbol
  text = text.replace(/(?:\$|\\\$)+([^\n$]+?)(?:\\*\$)+/g, (_, math) => {
    let clean = math.replace(/\\/g, "").trim()
    if (!clean) return ""
    if (clean === "rho") clean = "ρ"
    if (clean === "pi") clean = "π"
    return `**${clean}**`
  })

  // 7. Clean up any remaining stray backslashes before English letters or math signs (e.g. \rho, \$)
  text = text.replace(/\\([a-zA-Z]+)/g, (_, word) => {
    if (word === "rho") return "ρ"
    if (word === "pi") return "π"
    return word
  })
  text = text.replace(/\\([$()\[\]{}*\/=+\-])/g, "$1")

  // 8. Fix floating/lone bullets followed by newlines
  text = text.replace(/(?:^|\n)\s*[•\-\*]\s*\n+(?=[^\n])/g, "\n• ")
  text = text.replace(/(?:^|\n)\s*[•\-\*]\s*(?=\n|$)/g, "")

  // 9. Normalize spaces around division slashes between digits and clean excessive spaces
  text = text.replace(/(\d+)\s*\/\s*(\d+)/g, "$1 / $2")
  text = text.replace(/[ \t]{2,}/g, " ")

  return text
}

/**
 * Preprocess and structure Persian educational textbook response text for Markdown rendering.
 * Converts raw OCR fragments, tables, and activities into structured GitHub-Flavored Markdown.
 */
function formatPersianScientificMarkdown(raw: string): string {
  if (!raw) return ""

  // 0. Clean and sanitize any raw LaTeX formulas or broken bullets
  let text = cleanLatexAndFormulas(raw)

  // 0.0 Strip any internal memory summary tags, instructions, or metadata markers so they NEVER show to the user
  text = text.replace(/(?:دستورالعمل ثبت خلاصه[^\n]*\n+)?<!--\s*SUMMARY\s*-->[\s\S]*?(?:<!--\s*\/SUMMARY\s*-->|$)/gi, "")
  text = text.replace(/<!--[\s\S]*?-->/g, "")
  text = text.replace(/دستورالعمل ثبت خلاصه در حافظه گفت[‌و]?وگو:?[\s\S]*?(?=(?:\n\s*---\s*)?\n+\s*(?:\*{0,2}:?Sources:?\*{0,2}|####?\s*📚)|$)/gi, "")
  text = text.replace(/توجه بسیار مهم:\s*از خلاصه‌سازی[\s\S]*?(?=(?:\n\s*---\s*)?\n+\s*(?:\*{0,2}:?Sources:?\*{0,2}|####?\s*📚)|$)/gi, "")
  text = text.trim()

  // 0.1 Pre-heal any markdown links where spaces or chapter names were introduced inside URL target
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (fullMatch, linkText, rawTarget) => {
    if (rawTarget.includes("pdf") || rawTarget.includes("api") || rawTarget.includes("فصل") || rawTarget.includes("olom_haftom")) {
      const fixedUrl = resolvePdfUrl(rawTarget, linkText)
      return `[${linkText}](${fixedUrl})`
    }
    return fullMatch
  })

  // 1. Clean messy raw filename mentions in plain text, but NEVER touch URLs inside (...)
  text = text.replace(/(\[[^\]]*\]\([^)]+\))|(C706_olom_haftom[^\s#?)]+\.pdf)(?:\s*[-–—]\s*(?:pages?|صفحات?|صفحه)?\s*([0-9,\s]+))?/gi, (match, link, file, pages) => {
    if (link) return link // Preserve markdown links completely untouched!
    return formatChapterCitationLabel(file, pages ? pages.trim() : "")
  })

  // 2. Convert bullet headers like "• (صفحه 6): ### فعالیت ..." or "• (صفحه 6): ..." into clean headings
  text = text.replace(/•\s*\(صفحه\s*(\d+)\)\s*:\s*###\s*(.*)/g, "\n\n### 📄 صفحه $1: $2\n")
  text = text.replace(/•\s*\(صفحه\s*(\d+)\)\s*:\s*(.*)/g, "\n\n### 📄 صفحه $1\n$2")

  // 3. Add visual icons to curriculum section headings
  text = text.replace(/###\s*(فعالیت[^\n]*)/g, "### 📝 $1")
  text = text.replace(/###\s*(فکر کنید[^\n]*)/g, "### 💡 $1")
  text = text.replace(/###\s*(آزمایش کنید[^\n]*)/g, "### 🧪 $1")
  text = text.replace(/###\s*(اطلاعات جمع‌آوری کنید[^\n]*)/g, "### 🔍 $1")

  // 4. Convert multi-line plain text table (e.g. ویژگی / سرخرگ / سیاهرگ) into standard Markdown table
  const tableHeaderRegex = /(?:ویژگی\s*\n\s*سرخرگ\s*\n\s*سیاهرگ)([\s\S]*?)(?=\n\s*(?:###|•|Sources:|📚|\[\d+\]|$))/i
  const tableMatch = text.match(tableHeaderRegex)
  if (tableMatch) {
    const rawRows = tableMatch[1]
      .split("\n")
      .map((r) => r.trim())
      .filter(Boolean)

    if (rawRows.length > 0) {
      const tableLines = [
        "| ویژگی | سرخرگ | سیاهرگ |",
        "| :--- | :---: | :---: |",
      ]
      for (const row of rawRows) {
        let sorkh = "✔"
        let siah = "✔"
        if (row.includes("بطن")) {
          sorkh = "✔"
          siah = "—"
        } else if (row.includes("دهلیز")) {
          sorkh = "—"
          siah = "✔"
        } else if (row.includes("از قلب به اندام")) {
          sorkh = "✔"
          siah = "—"
        } else if (row.includes("از اندام") && row.includes("به قلب")) {
          sorkh = "—"
          siah = "✔"
        }
        tableLines.push(`| ${row} | ${sorkh} | ${siah} |`)
      }
      text = text.replace(tableMatch[0], "\n\n" + tableLines.join("\n") + "\n\n")
    }
  }

  // 5. Format bracketed cycle descriptions like "[نمودار چرخهای ...: چرخه در سه مرحله ...]"
  text = text.replace(/\[\s*(نمودار چرخ[^\:]*:[^\]]+)\]/g, (_, content) => {
    const lines = content.split(":")
    const title = lines[0] || "نمودار چرخه‌ای کار قلب"
    const rest = lines.slice(1).join(":\n> - ")
    return `\n\n> 🔄 **${title}:**\n> ${rest}\n\n`
  })

  // 6. Deduplicate & format Sources block at bottom with clean divider & clickable PDF links
  // If the text contains one or more source sections, isolate the main answer body and consolidate all unique citations
  const sourcesHeaderRegex = /(?:\n\s*---\s*)?\n+\s*(?:\*{0,2}:?Sources:?\*{0,2}|####?\s*📚\s*منابع استنادی کتاب درسی:?)/i
  if (sourcesHeaderRegex.test(text)) {
    const parts = text.split(sourcesHeaderRegex)
    const answerBody = parts[0].trim()
    const rawSources = parts.slice(1).join("\n")

    const citationRegex = /\[(\d+)\]\s*(?:\[([^\]]+)\]\(([^)]+)\)|\[?([^\s\]()]+\.pdf)\]?(?:\s*[-–—]\s*(?:pages?|صفحات?|صفحه)?\s*([0-9,\s]+))?)/g
    const uniqueCitations = new Map<string, { label: string; url: string }>()

    let m: RegExpExecArray | null
    while ((m = citationRegex.exec(rawSources)) !== null) {
      const [_, __, linkText, linkUrl, rawFile, rawPages] = m
      if (linkText && linkUrl) {
        const targetUrl = resolvePdfUrl(linkUrl, linkText)
        if (!uniqueCitations.has(targetUrl)) {
          uniqueCitations.set(targetUrl, { label: linkText, url: targetUrl })
        }
      } else if (rawFile) {
        const firstPage = rawPages ? rawPages.split(",")[0].trim() : "1"
        const label = formatChapterCitationLabel(rawFile, rawPages || firstPage)
        const targetUrl = resolvePdfUrl(`/api/v1/pdf/${rawFile}#page=${firstPage}`, label)
        if (!uniqueCitations.has(targetUrl)) {
          uniqueCitations.set(targetUrl, { label, url: targetUrl })
        }
      }
    }

    if (uniqueCitations.size > 0) {
      let counter = 1
      const lines: string[] = []
      Array.from(uniqueCitations.values()).forEach((cit) => {
        lines.push(`- **[${counter}] [${cit.label}](${cit.url})**`)
        counter++
      })
      text = `${answerBody}\n\n---\n\n#### 📚 منابع استنادی کتاب درسی:\n${lines.join("\n")}`
    } else {
      text = answerBody
    }
  }

  return text
}

export function ChatMessage({ message, isStreaming, streamingStatus, onSelectAsContext }: ChatMessageProps) {
  const [copied, setCopied] = useState(false)
  const [showTelemetry, setShowTelemetry] = useState(false)

  const isUser = message.role === "user"
  const formattedContent = isUser ? message.content : formatPersianScientificMarkdown(message.content)

  const handleCopy = () => {
    const cleanToCopy = message.content
      .replace(/(?:دستورالعمل ثبت خلاصه[^\n]*\n+)?<!--\s*SUMMARY\s*-->[\s\S]*?(?:<!--\s*\/SUMMARY\s*-->|$)/gi, "")
      .replace(/<!--[\s\S]*?-->/g, "")
      .replace(/دستورالعمل ثبت خلاصه در حافظه گفت[‌و]?وگو:?[\s\S]*?(?=(?:\n\s*---\s*)?\n+\s*(?:\*{0,2}:?Sources:?\*{0,2}|####?\s*📚)|$)/gi, "")
      .replace(/توجه بسیار مهم:\s*از خلاصه‌سازی[\s\S]*?(?=(?:\n\s*---\s*)?\n+\s*(?:\*{0,2}:?Sources:?\*{0,2}|####?\s*📚)|$)/gi, "")
      .trim()
    navigator.clipboard.writeText(cleanToCopy)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Format latency in clean seconds or ms
  const formatMs = (ms?: number) => {
    if (ms === undefined || ms === null) return "-"
    if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`
    return `${Math.round(ms)}ms`
  }

  return (
    <div
      className={`py-5 px-4 md:px-8 border-b border-border/40 transition-colors ${
        isUser ? "bg-background" : "bg-muted/15"
      }`}
    >
      <div className="max-w-3xl mx-auto flex gap-3.5 md:gap-4 items-start">
        {/* Avatar */}
        <div
          className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 text-xs font-semibold shadow-sm ${
            isUser
              ? "bg-secondary text-secondary-foreground border border-border"
              : "bg-primary text-primary-foreground"
          }`}
        >
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </div>

        {/* Message Content */}
        <div className="flex-1 min-w-0 space-y-3">
          {/* Header line: Role and Timestamp */}
          <div className="flex items-center justify-between text-xs text-muted-foreground select-none">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-foreground">
                {isUser ? "شما" : "دستیار علوم تجربی"}
              </span>
              <span className="text-[10px] text-muted-foreground/80">
                {new Date(message.timestamp).toLocaleTimeString("fa-IR", {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
              {!isUser && !message.is_refusal && message.query_rewritten && (
                <span className="bg-primary/10 text-primary text-[10px] px-1.5 py-0.5 rounded-md font-medium">
                  بازنویسی ضمایر اعمال شد
                </span>
              )}
              {isStreaming && streamingStatus && message.content && (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-primary/10 text-primary text-[11px] font-medium border border-primary/20 animate-in fade-in duration-200">
                  <ThinkingDots className="w-3.5 h-3.5" />
                  <span>{streamingStatus}</span>
                </span>
              )}
            </div>

            {/* Action buttons */}
            <div className="flex items-center gap-1.5 opacity-80 hover:opacity-100 transition-opacity">
              {onSelectAsContext && message.content && (
                <button
                  onClick={() => onSelectAsContext(message)}
                  className="flex items-center gap-1 px-2 py-0.5 rounded-md hover:bg-muted text-[11px] text-muted-foreground hover:text-primary transition-all border border-border/40"
                  title="انتخاب این پیام به عنوان بافت پرسش بعدی"
                >
                  <Reply className="w-3 h-3" />
                  <span>پاسخ / بافت</span>
                </button>
              )}
              <button
                onClick={handleCopy}
                className="p-1 hover:bg-muted rounded-md text-muted-foreground hover:text-foreground transition-all"
                title="رونویسی متن"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          {/* Conversational Context Badge (User Story 2 & 6) */}
          {!isUser && message.conversational_context_badge && (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-medium">
              <MessageSquare className="w-3.5 h-3.5 text-sky-400 shrink-0" />
              <span>{message.conversational_context_badge}</span>
            </div>
          )}

          {/* Refusal Notice Banner */}
          {message.is_refusal && (
            <div className="p-3 rounded-xl border border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400 text-xs flex items-center gap-2.5">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>
                {message.content && (message.content.includes("فصل‌های انتخابی") || message.content.includes("محدوده فصل"))
                  ? "این پرسش در محدوده فصل‌های انتخابی قرار ندارد و نیاز به تغییر دامنه جستجو دارد."
                  : "این پرسش خارج از دامنه کتاب علوم تجربی هفتم ارزیابی شد و برای حفظ دقت استنادی رد شد."}
              </span>
            </div>
          )}

          {/* Sleek Thinking State (1:1 with user's screenshot) */}
          {!isUser && (!message.content || message.content.length === 0) && (
            <div className="flex items-center gap-3 py-3 px-1 text-sm md:text-base font-normal text-foreground/90 select-none animate-in fade-in duration-300">
              {isStreaming ? (
                <>
                  <ThinkingDots className="w-4 h-4 text-primary" />
                  <span className="tracking-tight font-medium">
                    {streamingStatus || message.status_title || "تحلیل و بررسی سوال..."}
                  </span>
                </>
              ) : (
                <div className="flex items-center gap-2 text-destructive text-sm font-medium">
                  <AlertCircle className="w-4 h-4" />
                  <span>پاسخی از سرور دریافت نشد یا ارتباط با مدل محلی قطع شد. لطفاً دوباره تلاش کنید.</span>
                </div>
              )}
            </div>
          )}

          {/* Body Text (Markdown with rich responsive components) */}
          {message.content && (
          <div className="markdown-content max-w-none text-foreground leading-relaxed break-words font-normal">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ node, ...props }) => (
                  <h1 className="text-xl font-bold text-foreground mt-4 mb-2 pb-1 border-b border-border/60" {...props} />
                ),
                h2: ({ node, ...props }) => (
                  <h2 className="text-lg font-bold text-foreground mt-3.5 mb-2" {...props} />
                ),
                h3: ({ node, ...props }) => (
                  <h3 className="text-sm md:text-base font-bold text-primary flex items-center gap-2 mt-4 mb-2.5 bg-primary/10 px-3 py-1.5 rounded-xl border border-primary/20" {...props} />
                ),
                h4: ({ node, ...props }) => (
                  <h4 className="text-xs md:text-sm font-bold text-foreground/90 mt-3 mb-1.5" {...props} />
                ),
                p: ({ node, ...props }) => (
                  <p className="mb-2.5 last:mb-0 leading-relaxed text-sm text-foreground/90 font-normal" {...props} />
                ),
                ul: ({ node, ...props }) => (
                  <ul className="list-disc list-inside space-y-1.5 my-2.5 text-sm text-foreground/90 pr-2" {...props} />
                ),
                ol: ({ node, ...props }) => (
                  <ol className="list-decimal list-inside space-y-1.5 my-2.5 text-sm text-foreground/90 pr-2" {...props} />
                ),
                li: ({ node, ...props }) => (
                  <li className="leading-relaxed" {...props} />
                ),
                blockquote: ({ node, ...props }) => (
                  <blockquote className="border-r-4 border-primary bg-primary/5 px-4 py-3 rounded-l-xl my-3 text-xs md:text-sm text-foreground/90 font-medium" {...props} />
                ),
                table: ({ node, ...props }) => (
                  <div className="overflow-x-auto my-3 rounded-xl border border-border bg-card shadow-xs">
                    <table className="w-full text-xs md:text-sm border-collapse text-right" {...props} />
                  </div>
                ),
                thead: ({ node, ...props }) => (
                  <thead className="bg-muted text-foreground font-bold border-b border-border" {...props} />
                ),
                tbody: ({ node, ...props }) => (
                  <tbody className="divide-y divide-border/40" {...props} />
                ),
                tr: ({ node, ...props }) => (
                  <tr className="hover:bg-muted/30 transition-colors" {...props} />
                ),
                th: ({ node, ...props }) => (
                  <th className="px-3.5 py-2.5 font-bold text-foreground text-right border-b border-border" {...props} />
                ),
                td: ({ node, ...props }) => (
                  <td className="px-3.5 py-2 text-foreground/90 text-right border-b border-border/30" {...props} />
                ),
                code: ({ node, className, children, ...props }) => {
                  const isInline = !className?.includes("language-")
                  return isInline ? (
                    <code className="bg-muted/80 text-primary font-mono text-[11px] px-1.5 py-0.5 rounded border border-border/50 font-semibold" {...props}>
                      {children}
                    </code>
                  ) : (
                    <pre className="bg-muted/60 p-3 rounded-xl overflow-x-auto text-xs font-mono my-2.5 border border-border">
                      <code {...props}>{children}</code>
                    </pre>
                  )
                },
                hr: ({ node, ...props }) => (
                  <hr className="my-3.5 border-border/60" {...props} />
                ),
                a: ({ node, href, children, ...props }) => {
                  const labelStr = typeof children === "string" ? children : Array.isArray(children) ? children.join(" ") : ""
                  const finalHref = resolvePdfUrl(href || "#", labelStr)
                  let linkText = children

                  if (typeof children === "string" && (children.includes("C706") || children.endsWith(".pdf"))) {
                    const pageMatch = finalHref.match(/page=(\d+)/)
                    linkText = formatChapterCitationLabel(children, pageMatch ? pageMatch[1] : "")
                  }

                  return (
                    <a
                      href={finalHref}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-primary hover:underline font-medium text-xs bg-primary/10 px-2 py-0.5 rounded-md border border-primary/20 transition-colors mx-0.5 hover:bg-primary/20"
                      {...props}
                    >
                      <span>{linkText}</span>
                      <ExternalLink className="w-3 h-3 inline shrink-0" />
                    </a>
                  )
                },
                strong: ({ node, ...props }) => (
                  <strong className="font-bold text-foreground" {...props} />
                ),
              }}
            >
              {formattedContent}
            </ReactMarkdown>
          </div>
          )}

          {/* Telemetry Waterfall Badge (Our unique feature!) */}
          {!isUser && message.latency_profile && (
            <div className="pt-1">

              <button
                onClick={() => setShowTelemetry(!showTelemetry)}
                className="inline-flex items-center gap-1.5 text-[10px] font-mono text-muted-foreground hover:text-foreground bg-muted/30 px-2 py-0.5 rounded-md hover:bg-muted transition-colors"
              >
                <Zap className="w-3 h-3 text-amber-500" />
                <span>
                  زمان کل: {formatMs(message.latency_profile.total_ms)}
                  {message.latency_profile.queue_wait_ms !== undefined &&
                    ` (صف: ${formatMs(message.latency_profile.queue_wait_ms)})`}
                </span>
                {showTelemetry ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>

              {/* Waterfall Breakdown Drawer */}
              {showTelemetry && (
                <div className="mt-2 p-2.5 rounded-xl bg-card border border-border text-[11px] space-y-1 font-mono animate-in fade-in duration-200">
                  <div className="text-muted-foreground font-semibold text-[10px] pb-1 border-b border-border/40">
                    آبشار تاخیر مراحل پردازش (Latency Waterfall)
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 pt-1 text-muted-foreground">
                    <div className="flex justify-between">
                      <span>صف همزمانی:</span>
                      <span className="text-foreground">{formatMs(message.latency_profile.queue_wait_ms)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>تصمیم/بازنویسی Jev:</span>
                      <span className="text-foreground">{formatMs(message.latency_profile.rewrite_ms)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>بازیابی ترکیبی:</span>
                      <span className="text-foreground">{formatMs(message.latency_profile.retrieval_ms)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>بازرتبه‌بندی عصبی:</span>
                      <span className="text-foreground">{formatMs(message.latency_profile.rerank_ms)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>بسط بخش والد:</span>
                      <span className="text-foreground">{formatMs(message.latency_profile.expansion_ms)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>تولید مدل زبانی:</span>
                      <span className="text-foreground">{formatMs(message.latency_profile.synthesis_ms)}</span>
                    </div>
                  </div>

                  {/* Capacity Planning & Budgeting (Cost / 1k Q) */}
                  <div className="pt-2 mt-2 border-t border-border/40 space-y-1">
                    <div className="flex items-center justify-between text-muted-foreground font-semibold text-[10px]">
                      <span className="flex items-center gap-1 text-emerald-500">
                        <DollarSign className="w-3 h-3" />
                        برنامه‌ریزی ظرفیت و بودجه (هزینه ۱۰۰۰ پرسش مشابه)
                      </span>
                      {message.cost_1k && (
                        <span className="text-[10px] text-foreground font-bold font-mono">
                          ${message.cost_1k.min_1k_usd.toFixed(2)} – ${message.cost_1k.max_1k_usd.toFixed(2)}
                        </span>
                      )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1 text-[10px]">
                      {/* Min Model */}
                      <div className="p-1.5 rounded-lg bg-muted/40 border border-border/40">
                        <div className="flex justify-between items-center text-foreground font-semibold">
                          <span>مدل اقتصادی (Min):</span>
                          <span className="text-emerald-500 font-bold font-mono">
                            ${message.cost_1k?.min_1k_usd.toFixed(2) || "0.55"} / 1k Q
                          </span>
                        </div>
                        <div className="text-[9px] text-muted-foreground mt-0.5">
                          ورودی: $0.15/1M (کش: $0.003/1M) | خروجی: $0.60/1M
                        </div>
                      </div>

                      {/* Max Model */}
                      <div className="p-1.5 rounded-lg bg-muted/40 border border-border/40">
                        <div className="flex justify-between items-center text-foreground font-semibold">
                          <span>مدل پیشرفته (Max):</span>
                          <span className="text-purple-400 font-bold font-mono">
                            ${message.cost_1k?.max_1k_usd.toFixed(2) || "6.00"} / 1k Q
                          </span>
                        </div>
                        <div className="text-[9px] text-muted-foreground mt-0.5">
                          ورودی: $1.50/1M | خروجی: $7.50/1M | کش ورودی رایگان
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Langfuse Trace Link */}
                  <div className="pt-1.5 flex items-center justify-between text-[10px] text-muted-foreground border-t border-border/30">
                    <span>ردیابی لنگ‌فیوز (Langfuse Trace):</span>
                    {message.trace_id && message.trace_id !== "offline_trace" ? (
                      <a
                        href={`https://cloud.langfuse.com/project/cmuyenvm200psad0ml6jrdfhr/traces/${message.trace_id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-primary hover:underline font-mono"
                      >
                        <span>{message.trace_id.slice(0, 10)}...</span>
                        <ExternalLink className="w-2.5 h-2.5" />
                      </a>
                    ) : (
                      <a
                        href="https://cloud.langfuse.com/project/cmuyenvm200psad0ml6jrdfhr/traces"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-primary hover:underline font-mono"
                      >
                        <span>مشاهده لاگ پروژه</span>
                        <ExternalLink className="w-2.5 h-2.5" />
                      </a>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
