"use client"

import React, { useState, useEffect } from "react"
import { OLUM_7_CHAPTERS } from "@/lib/constants"
import { X, Check, BookOpen, Layers, HelpCircle, Info } from "lucide-react"

interface ChapterModalProps {
  isOpen: boolean
  onClose: () => void
  selectedChapters: number[]
  onSaveChapters: (chapters: number[]) => void
  selectedSources?: string[]
  onToggleSource?: (source: "science_textbook" | "administrative_faq") => void
  pipelineMode?: "advanced" | "naive"
}

export function ChapterModal({
  isOpen,
  onClose,
  selectedChapters,
  onSaveChapters,
  selectedSources = ["science_textbook"],
  onToggleSource,
  pipelineMode = "advanced",
}: ChapterModalProps) {
  const [tempSelected, setTempSelected] = useState<number[]>(selectedChapters)
  const isEnterprise = pipelineMode === "advanced"

  const isScienceActive = selectedSources.includes("science_textbook")
  const isFaqActive = selectedSources.includes("administrative_faq")

  const [activeTab, setActiveTab] = useState<"science" | "faq">("science")

  useEffect(() => {
    setTempSelected(selectedChapters)
    if (!isScienceActive && isFaqActive) {
      setActiveTab("faq")
    } else {
      setActiveTab("science")
    }
  }, [selectedChapters, selectedSources, isOpen, isScienceActive, isFaqActive])

  if (!isOpen) return null

  const toggleChapter = (id: number) => {
    if (tempSelected.includes(id)) {
      setTempSelected(tempSelected.filter((c) => c !== id))
    } else {
      setTempSelected([...tempSelected, id].sort((a, b) => a - b))
    }
  }

  const handleSelectAll = () => {
    setTempSelected(OLUM_7_CHAPTERS.map((c) => c.id))
  }

  const handleClearAll = () => {
    setTempSelected([])
  }

  const handleSave = () => {
    onSaveChapters(tempSelected)
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-card border border-border rounded-2xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl animate-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="p-4 border-b border-border/80 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-foreground">
                {isEnterprise && selectedSources.length > 1
                  ? "مدیریت دامنه و منابع دانش (کتاب علوم + پرسش‌های متداول)"
                  : activeTab === "faq"
                  ? "دامنه پرسش‌های متداول صندوق نوآوری (FAQ)"
                  : "انتخاب فصل‌های کتاب علوم تجربی هفتم"}
              </h2>
              <p className="text-[11px] text-muted-foreground">
                {isEnterprise
                  ? "پیکربندی دامنه سرفصل‌ها و فعال‌سازی منابع برای دستیار هوشمند پیشرفته"
                  : "فصل‌های مورد نظر برای تمرین هدفمند را تیک بزنید؛ در صورت خالی بودن، جستجو در کل کتاب انجام می‌شود."}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-muted text-muted-foreground hover:text-foreground rounded-lg transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Enterprise Source Switcher Tabs inside Modal */}
        {isEnterprise && (
          <div className="px-4 pt-2.5 pb-0 bg-muted/40 border-b border-border/60 flex items-center gap-2">
            <button
              onClick={() => setActiveTab("science")}
              className={`px-3 py-2 rounded-t-xl text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
                activeTab === "science"
                  ? "border-primary text-primary bg-background shadow-xs"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:bg-muted/30"
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>کتاب علوم هفتم (۱۵ فصل)</span>
              {isScienceActive && (
                <span className="w-2 h-2 rounded-full bg-primary inline-block" title="منبع فعال است" />
              )}
            </button>

            <button
              onClick={() => setActiveTab("faq")}
              className={`px-3 py-2 rounded-t-xl text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
                activeTab === "faq"
                  ? "border-emerald-500 text-emerald-600 dark:text-emerald-400 bg-background shadow-xs"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:bg-muted/30"
              }`}
            >
              <HelpCircle className="w-3.5 h-3.5" />
              <span>پرسش‌های متداول صندوق (۱۳۴ مورد)</span>
              {isFaqActive && (
                <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" title="منبع فعال است" />
              )}
            </button>
          </div>
        )}

        {/* Tab 1: Science Textbook Chapters Grid */}
        {activeTab === "science" && (
          <>
            {/* Quick actions & source activation bar */}
            <div className="px-4 py-2 bg-muted/30 border-b border-border/60 flex items-center justify-between text-xs select-none">
              <span className="text-[11px] text-muted-foreground">
                {tempSelected.length === 0
                  ? "حالت فعلی: جستجوی سراسری در تمام ۱۵ فصل"
                  : `${tempSelected.length} فصل انتخاب شده`}
              </span>
              <div className="flex items-center gap-3">
                {isEnterprise && onToggleSource && (
                  <button
                    type="button"
                    onClick={() => onToggleSource("science_textbook")}
                    className={`px-2 py-0.5 rounded-md text-[10px] font-medium border transition-all ${
                      isScienceActive
                        ? "bg-primary/15 text-primary border-primary/30"
                        : "bg-muted text-muted-foreground hover:text-foreground border-border"
                    }`}
                  >
                    {isScienceActive ? "✓ فعال در گفتگو" : "+ فعال‌سازی منبع"}
                  </button>
                )}
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleSelectAll}
                    className="text-primary hover:underline text-[11px] font-medium"
                  >
                    انتخاب همه
                  </button>
                  <span className="text-muted-foreground">•</span>
                  <button
                    onClick={handleClearAll}
                    className="text-muted-foreground hover:text-foreground text-[11px]"
                  >
                    پاک کردن همه
                  </button>
                </div>
              </div>
            </div>

            {/* Chapters Grid */}
            <div className="flex-1 overflow-y-auto p-4 grid grid-cols-1 md:grid-cols-2 gap-2.5">
              {OLUM_7_CHAPTERS.map((ch) => {
                const isChecked = tempSelected.includes(ch.id)
                return (
                  <div
                    key={ch.id}
                    onClick={() => toggleChapter(ch.id)}
                    className={`p-3 rounded-xl border cursor-pointer transition-all flex items-start gap-3 ${
                      isChecked
                        ? "bg-primary/10 border-primary/60 text-foreground shadow-sm"
                        : "bg-muted/20 border-border/70 hover:bg-muted/50 text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <div
                      className={`w-5 h-5 rounded-md border flex items-center justify-center shrink-0 mt-0.5 transition-all ${
                        isChecked
                          ? "bg-primary border-primary text-primary-foreground"
                          : "border-muted-foreground/40 bg-background"
                      }`}
                    >
                      {isChecked && <Check className="w-3.5 h-3.5 stroke-[3]" />}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-xs font-bold leading-snug">{ch.title}</div>
                      <div className="text-[10px] text-muted-foreground leading-relaxed mt-0.5">
                        {ch.subtitle}
                      </div>
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {ch.topics.map((t, idx) => (
                          <span
                            key={idx}
                            className="text-[9px] bg-background/60 text-muted-foreground px-1.5 py-0.2 rounded border border-border/40 font-mono"
                          >
                            #{t}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </>
        )}

        {/* Tab 2: Administrative FAQ Source (No parent selection needed) */}
        {activeTab === "faq" && (
          <div className="flex-1 overflow-y-auto p-5 space-y-4 select-none">
            <div className="p-4 rounded-2xl border border-emerald-500/30 bg-emerald-500/5 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 flex items-center justify-center font-bold text-sm">
                    FAQ
                  </div>
                  <div>
                    <h3 className="text-xs md:text-sm font-bold text-foreground">
                      پرسش‌های متداول صندوق نوآوری و شکوفایی (INIF)
                    </h3>
                    <p className="text-[10px] md:text-[11px] text-muted-foreground">
                      ۱۳۴ پرسش و پاسخ رسمی صندوق در موضوعات تسهیلات، ضمانت‌نامه، لیزینگ و سرمایه‌گذاری
                    </p>
                  </div>
                </div>

                {isEnterprise && onToggleSource && (
                  <button
                    type="button"
                    onClick={() => onToggleSource("administrative_faq")}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
                      isFaqActive
                        ? "bg-emerald-600 text-white border-emerald-600 shadow-xs"
                        : "bg-muted text-muted-foreground hover:text-foreground border-border"
                    }`}
                  >
                    {isFaqActive ? "✓ منبع فعال است" : "+ فعال‌سازی این منبع"}
                  </button>
                )}
              </div>

              <div className="text-xs text-muted-foreground leading-relaxed pt-2 border-t border-emerald-500/20 space-y-2">
                <p className="text-[11px]">
                  این منبع پاسخ‌های مصوب صندوق نوآوری به سوالات شرکت‌های دانش‌بنیان و فناور را پوشش می‌دهد.
                </p>
                <div className="p-2.5 rounded-xl bg-background/80 border border-emerald-500/25 text-[11px] text-emerald-700 dark:text-emerald-300 flex items-center gap-2">
                  <Info className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
                  <span>
                    بدون نیاز به انتخاب سرفصل والد: جستجوی متنی و معنایی مستقیماً در کل بانک ۱۳۴ موردی انجام می‌شود.
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="p-3.5 border-t border-border/80 flex items-center justify-end gap-2 bg-muted/20">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          >
            انصراف
          </button>
          <button
            onClick={handleSave}
            className="px-5 py-2 rounded-xl text-xs font-semibold bg-primary text-primary-foreground hover:opacity-95 shadow-sm transition-all"
          >
            اعمال تنظیمات
          </button>
        </div>
      </div>
    </div>
  )
}

