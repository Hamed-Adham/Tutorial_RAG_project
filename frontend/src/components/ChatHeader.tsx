"use client"

import React, { useState } from "react"
import { Activity, Sliders, BookOpen, Trash2, HelpCircle, Layers, Scale, Sparkles, Database } from "lucide-react"
import { Session, SystemHealth } from "@/types"

interface ChatHeaderProps {
  session: Session | null
  selectedChapters: number[]
  selectedCategories?: string[]
  onOpenChapterModal: () => void
  onOpenFAQModal?: () => void
  onOpenScopeModal?: () => void
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
  pipelineMode?: "advanced" | "naive"
  onChangePipelineMode?: (mode: "advanced" | "naive") => void
  source?: "science_textbook" | "administrative_faq" | "all"
  onChangeSource?: (source: "science_textbook" | "administrative_faq" | "all") => void
  selectedSources?: string[]
  onToggleSource?: (source: "science_textbook" | "administrative_faq") => void
}

export function ChatHeader({
  session,
  selectedChapters,
  selectedCategories = [],
  onOpenChapterModal,
  onOpenFAQModal,
  onOpenScopeModal,
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
  pipelineMode = "advanced",
  onChangePipelineMode,
  source = "science_textbook",
  onChangeSource,
  selectedSources,
  onToggleSource,
}: ChatHeaderProps) {
  const [showSettings, setShowSettings] = useState(false)

  const activeSources = selectedSources && selectedSources.length > 0
    ? selectedSources
    : source === "all"
    ? ["science_textbook", "administrative_faq"]
    : [source]

  const isScienceActive = activeSources.includes("science_textbook")
  const isFaqActive = activeSources.includes("administrative_faq")
  const isAll = isScienceActive && isFaqActive
  const isFAQ = !isScienceActive && isFaqActive
  const isNaive = pipelineMode === "naive"

  const handleToggle = (src: "science_textbook" | "administrative_faq") => {
    if (onToggleSource) {
      onToggleSource(src)
    } else if (onChangeSource) {
      if (!isNaive) {
        if (src === "science_textbook") {
          onChangeSource(isFaqActive ? (isScienceActive ? "administrative_faq" : "all") : "science_textbook")
        } else {
          onChangeSource(isScienceActive ? (isFaqActive ? "science_textbook" : "all") : "administrative_faq")
        }
      } else {
        onChangeSource(src)
      }
    }
  }

  const handleOpenScope = onOpenScopeModal || onOpenChapterModal

  return (
    <header className="h-14 border-b border-border/80 bg-background/80 backdrop-blur-md px-3 md:px-4 flex items-center justify-between z-10 select-none">
      {/* Left: Branding & Active Scope */}
      <div className="flex items-center gap-3 overflow-hidden">
        {/* Dynamic Distinct Emblem Badge */}
        {isAll ? (
          <div className="w-8 h-8 rounded-lg bg-indigo-500/15 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30 flex items-center justify-center font-bold text-xs shrink-0 shadow-xs">
            ALL
          </div>
        ) : isFAQ ? (
          <div className="w-8 h-8 rounded-lg bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 flex items-center justify-center font-bold text-xs shrink-0 shadow-xs">
            FAQ
          </div>
        ) : (
          <div className="w-8 h-8 rounded-lg bg-primary/10 text-primary border border-primary/20 flex items-center justify-center font-bold text-xs shrink-0 shadow-xs">
            ۷م
          </div>
        )}

        <div className="flex flex-col truncate">
          <div className="text-xs md:text-sm font-semibold truncate flex items-center gap-2">
            <span>
              {isAll
                ? "دستیار جامع چندمنبعی (کتاب علوم + پرسش‌های متداول)"
                : isFAQ
                ? "پرسش‌های متداول صندوق نوآوری و شکوفایی (INIF)"
                : session?.title || "دستیار هوشمند علوم تجربی هفتم"}
            </span>

            {/* Pipeline Mode Pill Badge */}
            <span
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded shrink-0 font-medium ${
                isNaive
                  ? "bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30"
                  : "bg-primary/10 text-primary border border-primary/20"
              }`}
            >
              {isNaive ? "پایپ‌لاین پایه (Naive)" : "پیشرفته (Enterprise)"}
            </span>
          </div>

          {/* Scope Selector Link */}
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground truncate">
            {isAll ? (
              <button
                onClick={handleOpenScope}
                className="hover:text-primary transition-colors flex items-center gap-1.5 font-medium text-foreground truncate"
                title="مشاهده و مدیریت دامنه منابع فعال"
              >
                <span className="flex items-center gap-1 text-primary/90">
                  <BookOpen className="w-3 h-3 text-primary" />
                  <span>{selectedChapters.length === 0 ? "دامنه: کل کتاب (۱۵ فصل)" : `دامنه: ${selectedChapters.length} فصل`}</span>
                </span>
                <span className="text-muted-foreground/50 font-bold">+</span>
                <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
                  <HelpCircle className="w-3 h-3" />
                  <span>پرسش‌های متداول (۱۳۴ مورد)</span>
                </span>
              </button>
            ) : isFAQ ? (
              <button
                onClick={handleOpenScope}
                className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors flex items-center gap-1 font-medium text-emerald-600/90 dark:text-emerald-400/90"
                title="مشاهده اطلاعات منبع پرسش‌های متداول"
              >
                <HelpCircle className="w-3 h-3" />
                <span>دامنه: کل بانک پرسش‌های متداول (۱۳۴ مورد)</span>
              </button>
            ) : (
              <button
                onClick={handleOpenScope}
                className="hover:text-primary transition-colors flex items-center gap-1 font-medium text-primary/90"
                title="انتخاب و تغییر سرفصل‌های کتاب علوم"
              >
                <BookOpen className="w-3 h-3" />
                <span>
                  {selectedChapters.length === 0
                    ? "دامنه: کل کتاب (۱۵ فصل)"
                    : `دامنه: فصل‌های ${selectedChapters.join("، ")}`}
                </span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Center/Right Controls */}
      <div className="flex items-center gap-2">
        {/* Source Switcher Toggle - Multi-toggle in Enterprise mode */}
        {(onToggleSource || onChangeSource) && (
          <div className="hidden lg:flex items-center bg-muted/60 p-0.5 rounded-lg border border-border/80 text-[11px]">
            <button
              onClick={() => handleToggle("science_textbook")}
              className={`px-2 py-1 rounded-md transition-all flex items-center gap-1.5 font-medium ${
                isScienceActive
                  ? "bg-primary text-primary-foreground shadow-xs font-semibold"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/40"
              }`}
              title="تغییر منبع دانش به کتاب علوم هفتم"
            >
              <BookOpen className="w-3 h-3" />
              <span>کتاب علوم هفتم</span>
            </button>
            <button
              onClick={() => handleToggle("administrative_faq")}
              className={`px-2 py-1 rounded-md transition-all flex items-center gap-1.5 font-medium ${
                isFaqActive
                  ? "bg-emerald-600 text-white shadow-xs font-semibold"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/40"
              }`}
              title="تغییر منبع دانش به پرسش‌های متداول (FAQ)"
            >
              <HelpCircle className="w-3 h-3" />
              <span>پرسش‌های متداول</span>
            </button>
          </div>
        )}

        {/* Pipeline Mode Toggle (Advanced vs Naive) */}
        {onChangePipelineMode && (
          <div className="flex items-center bg-muted/60 p-0.5 rounded-lg border border-border/80 text-[11px]">
            <button
              onClick={() => onChangePipelineMode("advanced")}
              className={`px-2 py-1 rounded-md transition-all flex items-center gap-1 font-medium ${
                !isNaive
                  ? "bg-primary text-primary-foreground shadow-xs font-semibold"
                  : "text-muted-foreground hover:text-foreground"
              }`}
              title="اجرا با پایپ‌لاین پیشرفته (بازنویسی پرسش، گراف دانش، رتبه‌بندی مجدد)"
            >
              <Sparkles className="w-3 h-3" />
              <span className="hidden sm:inline">پیشرفته</span>
            </button>
            <button
              onClick={() => onChangePipelineMode("naive")}
              className={`px-2 py-1 rounded-md transition-all flex items-center gap-1 font-medium ${
                isNaive
                  ? "bg-amber-600 text-white shadow-xs font-semibold"
                  : "text-muted-foreground hover:text-foreground"
              }`}
              title="اجرا با پایپ‌لاین پایه مستقیم (Cosine Similarity روی وکتور دیتابیس)"
            >
              <Database className="w-3 h-3" />
              <span className="hidden sm:inline">پایه</span>
            </button>
          </div>
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
            title="تنظیمات مدل و منبع داده"
          >
            <Sliders className="w-4 h-4" />
          </button>

          {showSettings && (
            <div className="absolute left-0 mt-2 w-72 bg-card border border-border rounded-xl shadow-xl p-3 z-30 text-xs space-y-3.5">
              <div className="font-semibold text-[11px] text-muted-foreground border-b border-border/60 pb-1.5 flex items-center justify-between">
                <span>تنظیمات و معماری پایپ‌لاین</span>
                <span className="text-[10px] font-mono">Config</span>
              </div>

              {/* Informative summary card of currently active pipeline mode */}
              <div className="p-2.5 rounded-xl border border-border/80 bg-muted/30 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs flex items-center gap-1.5">
                    {isNaive ? (
                      <>
                        <Database className="w-3.5 h-3.5 text-amber-500" />
                        <span className="text-amber-600 dark:text-amber-400">پایپ‌لاین پایه (Naive Baseline)</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5 text-primary" />
                        <span className="text-primary">پایپ‌لاین پیشرفته (Advanced Enterprise)</span>
                      </>
                    )}
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                    {isNaive ? "2 Stages" : "5 Stages"}
                  </span>
                </div>
                <p className="text-[10px] text-muted-foreground leading-relaxed">
                  {isNaive
                    ? "بازیابی تک‌مرحله‌ای مستقیم از دیتابیس برداری Qdrant با شباهت کسینوسی برداری و سنتز زبان طبیعی بدون بازنویسی یا بسط."
                    : "ارکستراسیون هوشمند ۵ مرحله‌ای: فیلتر معنایی با Jev، حافظه محاوره‌ای با SubQ، بازیابی ترکیبی BM25 + Qdrant، رتبه‌بندی مجدد BGE-Reranker و بسط بخش‌های کتاب."}
                </p>
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

              {/* History Toggle (only available in Advanced mode) */}
              {onToggleHistory && (
                <div className="flex items-center justify-between">
                  <div className="flex flex-col">
                    <span className="font-medium text-xs">حافظه گفت‌وگو</span>
                    <span className="text-[10px] text-muted-foreground">
                      {isNaive ? "در حالت پایه غیرفعال است" : "ارسال تاریخچه جلسات پیشین"}
                    </span>
                  </div>
                  <input
                    type="checkbox"
                    checked={!isNaive && historyEnabled}
                    disabled={isNaive}
                    onChange={onToggleHistory}
                    className="w-4 h-4 accent-primary cursor-pointer disabled:opacity-40"
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

              {/* Engine Status info - Synced per active source row */}
              <div className="pt-1.5 border-t border-border/60 text-[10px] space-y-2 text-muted-foreground">
                <div className="flex justify-between items-center">
                  <span>حالت فعال:</span>
                  <span className="font-mono text-foreground">
                    {isNaive ? "Naive Baseline (Direct Vector)" : "Enterprise LangGraph RAG"}
                  </span>
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between items-center">
                    <span>مجموعه داده:</span>
                    {isAll && (
                      <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-primary/10 text-primary border border-primary/20">
                        چندمنبعی (Multi-Corpus)
                      </span>
                    )}
                  </div>
                  <div className="space-y-1 pr-2 border-r-2 border-primary/50 text-[10px]">
                    {isScienceActive && (
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">کتاب علوم هفتم:</span>
                        <span className="font-mono text-foreground">science_curriculum (15 Ch)</span>
                      </div>
                    )}
                    {isFaqActive && (
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">پرسش‌های متداول:</span>
                        <span className="font-mono text-foreground">administrative_faq (134 QA)</span>
                      </div>
                    )}
                  </div>
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
