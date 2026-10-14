"use client"

import React, { useState, useRef, useEffect } from "react"
import { Send, Square, Sparkles, BookOpen, Reply, X } from "lucide-react"
import { SelectedContextTurn } from "@/types"

interface ChatInputProps {
  onSendMessage: (query: string) => void
  onStopStreaming: () => void
  isLoading: boolean
  isStreaming: boolean
  selectedChapters: number[]
  onOpenChapterModal: () => void
  selectedContextTurn?: SelectedContextTurn | null
  onClearSelectedContext?: () => void
}

const QUICK_SUGGESTIONS = [
    "مراحل اصلی حل مسئله به روش علمی چیست؟",
    "فرمول محاسبه چگالی چیست و یکای آن چیست؟",
    "تفاوت جرم و وزن چیست و با چه وسیله‌ای سنجیده می‌شود؟",
    "آبخوان و سفره آب زیرزمینی چگونه تشکیل می‌شود؟",
    "نقش میتوکندری در تولید انرژی یاخته چیست؟",
    "تبادل گازها در شش‌ها چگونه صورت می‌گیرد؟"
]

export function ChatInput({
  onSendMessage,
  onStopStreaming,
  isLoading,
  isStreaming,
  selectedChapters,
  onOpenChapterModal,
  selectedContextTurn,
  onClearSelectedContext,
}: ChatInputProps) {
  const [input, setInput] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
    }
  }, [input])

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (!input.trim() || isLoading) return
    onSendMessage(input.trim())
    setInput("")
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="p-4 bg-gradient-to-t from-background via-background/95 to-transparent border-t border-border/30">
      <div className="max-w-3xl mx-auto space-y-2.5">
        {/* Quick Suggestion Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs no-scrollbar select-none">
          <span className="text-[11px] text-muted-foreground flex items-center gap-1 shrink-0 px-1 font-medium">
            <Sparkles className="w-3 h-3 text-amber-500" />
            <span>پیشنهادها:</span>
          </span>
          {QUICK_SUGGESTIONS.map((sug, i) => (
            <button
              key={i}
              type="button"
              onClick={() => onSendMessage(sug)}
              className="text-[11px] bg-muted/60 hover:bg-muted text-foreground border border-border/60 hover:border-primary/50 px-2.5 py-1 rounded-full whitespace-nowrap transition-all shrink-0"
            >
              {sug}
            </button>
          ))}
        </div>

        {/* Floating Dismissible Context Turn Chip (User Story 6) */}
        {selectedContextTurn && (
          <div className="flex items-center justify-between px-3 py-1.5 rounded-xl bg-card border border-primary/40 shadow-sm text-xs text-foreground animate-in fade-in slide-in-from-bottom-2 duration-200">
            <div className="flex items-center gap-2 truncate">
              <span className="flex items-center gap-1 text-[11px] font-semibold text-primary shrink-0 bg-primary/10 px-2 py-0.5 rounded-md border border-primary/20">
                <Reply className="w-3 h-3 text-primary" />
                <span>بافت پیوست:</span>
              </span>
              <span className="truncate text-muted-foreground text-[11px]" dir="rtl">
                {selectedContextTurn.content.slice(0, 100)}...
              </span>
            </div>
            {onClearSelectedContext && (
              <button
                type="button"
                onClick={onClearSelectedContext}
                className="p-1 hover:bg-muted rounded-md text-muted-foreground hover:text-foreground transition-all shrink-0 ml-2"
                title="حذف بافت پیوست"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        )}

        {/* Input Box Wrapper */}
        <form
          onSubmit={handleSubmit}
          className="relative rounded-2xl border border-border/80 bg-card/90 shadow-lg focus-within:border-primary/60 focus-within:ring-2 focus-within:ring-primary/20 transition-all flex flex-col p-2"
        >
          {/* Active Scope Tag above text */}
          <div className="flex items-center justify-between px-2 pt-1 pb-1 text-[11px] text-muted-foreground border-b border-border/40 select-none">
            <button
              type="button"
              onClick={onOpenChapterModal}
              className="hover:text-primary transition-colors flex items-center gap-1 text-[11px] font-medium"
            >
              <BookOpen className="w-3 h-3 text-primary" />
              <span>
                {selectedChapters.length === 0
                  ? "دامنه جستجو: کل ۱۵ فصل کتاب (حالت عمومی)"
                  : `دامنه تمرین: فصل‌های ${selectedChapters.join("، ")}`}
              </span>
            </button>
            <span className="text-[10px] text-muted-foreground/80 hidden sm:inline">
              Enter برای ارسال • Shift+Enter برای خط جدید
            </span>
          </div>

          {/* Textarea */}
          <div className="flex items-end gap-2 pt-2">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="پرسش خود را درباره کتاب علوم تجربی هفتم بنویسید..."
              className="flex-1 bg-transparent border-0 focus:outline-none focus:ring-0 resize-none text-xs md:text-sm text-foreground placeholder:text-muted-foreground px-2 py-1 max-h-40 min-h-[40px] leading-relaxed"
              dir="rtl"
            />

            {/* Action Button: Send or Stop */}
            {isStreaming ? (
              <button
                type="button"
                onClick={onStopStreaming}
                className="w-8 h-8 rounded-xl bg-destructive text-destructive-foreground hover:opacity-90 flex items-center justify-center transition-all shrink-0 mb-0.5 shadow-sm"
                title="توقف تولید پاسخ"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className={`w-8 h-8 rounded-xl flex items-center justify-center transition-all shrink-0 mb-0.5 shadow-sm ${
                  input.trim() && !isLoading
                    ? "bg-primary text-primary-foreground hover:opacity-90"
                    : "bg-muted text-muted-foreground cursor-not-allowed opacity-60"
                }`}
                title="ارسال پرسش"
              >
                <Send className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  )
}
