"use client"

import React from "react"
import { OLUM_7_CHAPTERS } from "@/lib/constants"
import { X, Check, BookOpen, Layers } from "lucide-react"

interface ChapterModalProps {
  isOpen: boolean
  onClose: () => void
  selectedChapters: number[]
  onSaveChapters: (chapters: number[]) => void
}

export function ChapterModal({
  isOpen,
  onClose,
  selectedChapters,
  onSaveChapters,
}: ChapterModalProps) {
  const [tempSelected, setTempSelected] = React.useState<number[]>(selectedChapters)

  React.useEffect(() => {
    setTempSelected(selectedChapters)
  }, [selectedChapters, isOpen])

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
              <h2 className="text-sm font-bold">انتخاب فصل‌های کتاب علوم تجربی هفتم</h2>
              <p className="text-[11px] text-muted-foreground">
                فصل‌های مورد نظر برای تمرین هدفمند را تیک بزنید؛ در صورت خالی بودن، جستجو در کل کتاب انجام می‌شود.
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

        {/* Quick actions */}
        <div className="px-4 py-2 bg-muted/30 border-b border-border/60 flex items-center justify-between text-xs select-none">
          <span className="text-[11px] text-muted-foreground">
            {tempSelected.length === 0
              ? "حالت فعلی: جستجوی سراسری در تمام ۱۵ فصل"
              : `${tempSelected.length} فصل انتخاب شده`}
          </span>
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
