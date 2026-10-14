"use client"

import React, { useState } from "react"
import { Activity, Sliders, BookOpen, Trash2, Cpu, Zap, Wifi, Scale } from "lucide-react"
import { Session, SystemHealth } from "@/types"

interface ChatHeaderProps {
  session: Session | null
  selectedChapters: number[]
  onOpenChapterModal: () => void
  onOpenMetricsModal: () => void
  onOpenEvaluationModal?: () => void
  onClearSession: () => void
  isStreaming: boolean
  onToggleStreaming: () => void
  topK: number
  onChangeTopK: (k: number) => void
  health: SystemHealth | null
  historyEnabled?: boolean
  onToggleHistory?: () => void
}

export function ChatHeader({
  session,
  selectedChapters,
  onOpenChapterModal,
  onOpenMetricsModal,
  onOpenEvaluationModal,
  onClearSession,
  isStreaming,
  onToggleStreaming,
  topK,
  onChangeTopK,
  health,
  historyEnabled = true,
  onToggleHistory,
}: ChatHeaderProps) {
  const [showSettings, setShowSettings] = useState(false)

  return (
    <header className="h-14 border-b border-border/80 bg-background/80 backdrop-blur-md px-4 flex items-center justify-between z-10 select-none">
      {/* Session Title & Active Scope */}
      <div className="flex items-center gap-3 overflow-hidden">
        <div className="w-8 h-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center font-bold text-xs shrink-0">
          ۷م
        </div>
        <div className="flex flex-col truncate">
          <div className="text-xs md:text-sm font-semibold truncate flex items-center gap-2">
            <span>{session?.title || "دستیار هوشمند علوم تجربی هفتم"}</span>
            <span className="text-[10px] font-mono font-normal bg-muted px-1.5 py-0.5 rounded text-muted-foreground shrink-0">
              open-code
            </span>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground truncate">
            <button
              onClick={onOpenChapterModal}
              className="hover:text-primary transition-colors flex items-center gap-1 font-medium text-primary/90"
            >
              <BookOpen className="w-3 h-3" />
              <span>
                {selectedChapters.length === 0
                  ? "دامنه: کل کتاب (۱۵ فصل)"
                  : `دامنه: فصل‌های ${selectedChapters.join("، ")}`}
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-1.5">
        {/* Live Evaluation Suite Launcher Button */}
        {onOpenEvaluationModal && (
          <button
            onClick={onOpenEvaluationModal}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-primary/40 bg-primary/10 hover:bg-primary/20 text-xs font-semibold text-primary transition-all shadow-sm"
            title="اجرای ارزیابی، بنچ‌مارک‌ها و ماتریس تشخیصی"
          >
            <Scale className="w-3.5 h-3.5 text-primary" />
            <span className="hidden sm:inline text-[11px]">ارزیابی پایپ‌لاین</span>
          </button>
        )}

        {/* Live Metrics Trigger Button */}
        <button
          onClick={onOpenMetricsModal}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-border/80 hover:bg-muted/60 text-xs text-muted-foreground hover:text-foreground transition-all"
          title="مشاهده تله‌متری و سلامت همزمانی"
        >
          <Activity className="w-3.5 h-3.5 text-primary" />
          <span className="hidden sm:inline text-[11px] font-mono">تله‌متری</span>
        </button>

        {/* Quick Settings Dropdown Toggle */}
        <div className="relative">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-1.5 rounded-lg border border-border/80 hover:bg-muted/60 text-muted-foreground hover:text-foreground transition-all ${
              showSettings ? "bg-muted text-foreground" : ""
            }`}
            title="تنظیمات مدل و بازیابی"
          >
            <Sliders className="w-4 h-4" />
          </button>

          {showSettings && (
            <div className="absolute left-0 mt-2 w-64 bg-card border border-border rounded-xl shadow-xl p-3 z-30 text-xs space-y-3.5">
              <div className="font-semibold text-[11px] text-muted-foreground border-b border-border/60 pb-1.5 flex items-center justify-between">
                <span>تنظیمات پردازش</span>
                <span className="text-[10px] font-mono">Gateway Config</span>
              </div>

              {/* Streaming Toggle */}
              <div className="flex items-center justify-between">
                <div className="flex flex-col">
                  <span className="font-medium text-xs">پخش جریانی (SSE)</span>
                  <span className="text-[10px] text-muted-foreground">دریافت لحظه‌ای با TTFT &lt; ۱s</span>
                </div>
                <input
                  type="checkbox"
                  checked={isStreaming}
                  onChange={onToggleStreaming}
                  className="w-4 h-4 accent-primary cursor-pointer"
                />
              </div>

              {/* History / Conversational Memory Toggle */}
              {onToggleHistory && (
                <div className="flex items-center justify-between">
                  <div className="flex flex-col">
                    <span className="font-medium text-xs">حافظه گفت‌وگو</span>
                    <span className="text-[10px] text-muted-foreground">ارسال تاریخچه جلسات پیشین</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={historyEnabled}
                    onChange={onToggleHistory}
                    className="w-4 h-4 accent-primary cursor-pointer"
                  />
                </div>
              )}

              {/* Top-K Selector */}
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-muted-foreground">تعداد بندهای بازیابی (Top-K):</span>
                  <span className="font-mono font-bold text-primary">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={topK}
                  onChange={(e) => onChangeTopK(Number(e.target.value))}
                  className="w-full accent-primary cursor-pointer h-1.5 bg-muted rounded-lg"
                />
              </div>

              {/* Engine Status info */}
              <div className="pt-1.5 border-t border-border/60 text-[10px] space-y-1 text-muted-foreground">
                <div className="flex justify-between">
                  <span>موتور رتبه‌بندی:</span>
                  <span className="font-mono text-foreground">bge-reranker-v2-m3</span>
                </div>
                <div className="flex justify-between">
                  <span>بازیابی ترکیبی:</span>
                  <span className="font-mono text-foreground">Qdrant + BM25 (RRF)</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Clear session */}
        <button
          onClick={onClearSession}
          className="p-1.5 rounded-lg border border-border/80 hover:bg-destructive/10 hover:text-destructive text-muted-foreground transition-all"
          title="پاک‌کردن پیام‌های این گفتگو"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </header>
  )
}
