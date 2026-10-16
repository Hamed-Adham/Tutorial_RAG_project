"use client"

import React from "react"
import { FAQ_CATEGORIES } from "@/lib/constants"
import { X, Check, HelpCircle, Layers, Tag } from "lucide-react"

interface FAQCategoryModalProps {
  isOpen: boolean
  onClose: () => void
  selectedCategories: string[]
  onSaveCategories: (categories: string[]) => void
}

export function FAQCategoryModal({
  isOpen,
  onClose,
  selectedCategories,
  onSaveCategories,
}: FAQCategoryModalProps) {
  const [tempSelected, setTempSelected] = React.useState<string[]>(selectedCategories)

  React.useEffect(() => {
    setTempSelected(selectedCategories)
  }, [selectedCategories, isOpen])

  if (!isOpen) return null

  const toggleCategory = (id: string) => {
    if (tempSelected.includes(id)) {
      setTempSelected(tempSelected.filter((c) => c !== id))
    } else {
      setTempSelected([...tempSelected, id])
    }
  }

  const handleSelectAll = () => {
    setTempSelected(FAQ_CATEGORIES.map((c) => c.id))
  }

  const handleClearAll = () => {
    setTempSelected([])
  }

  const handleSave = () => {
    onSaveCategories(tempSelected)
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-card border border-border rounded-2xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl animate-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="p-4 border-b border-border/80 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center font-bold text-xs">
              FAQ
            </div>
            <div>
              <h2 className="text-sm font-bold text-foreground">دسته‌بندی‌های پرسش‌های متداول (صندوق نوآوری)</h2>
              <p className="text-[11px] text-muted-foreground">
                دسته‌های مورد نظر برای جستجوی متمرکز را انتخاب نمایید؛ در صورت خالی بودن، جستجو در تمام دسته‌ها انجام می‌شود.
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
              ? "حالت فعلی: جستجو در تمام ۹ دسته‌بندی (۱۳۴ پرسش)"
              : `${tempSelected.length} دسته انتخاب شده`}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={handleSelectAll}
              className="text-emerald-600 dark:text-emerald-400 hover:underline text-[11px] font-medium"
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

        {/* Categories Grid */}
        <div className="flex-1 overflow-y-auto p-4 grid grid-cols-1 md:grid-cols-2 gap-2.5">
          {FAQ_CATEGORIES.map((cat) => {
            const isChecked = tempSelected.includes(cat.id)
            return (
              <div
                key={cat.id}
                onClick={() => toggleCategory(cat.id)}
                className={`p-3 rounded-xl border text-right cursor-pointer transition-all flex flex-col justify-between ${
                  isChecked
                    ? "border-emerald-500/80 bg-emerald-500/5 shadow-sm"
                    : "border-border/70 hover:border-border hover:bg-muted/40"
                }`}
              >
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-4 h-4 rounded-md border flex items-center justify-center shrink-0 transition-colors ${
                        isChecked
                          ? "bg-emerald-600 border-emerald-600 text-white"
                          : "border-muted-foreground/40 bg-background"
                      }`}
                    >
                      {isChecked && <Check className="w-3 h-3 stroke-[3]" />}
                    </div>
                    <span className="text-xs font-semibold text-foreground line-clamp-1">
                      {cat.name}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 shrink-0">
                    {cat.count} پرسش
                  </span>
                </div>
                <p className="text-[10px] text-muted-foreground line-clamp-2 pr-6">
                  {cat.description}
                </p>
              </div>
            )
          })}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border/80 flex items-center justify-between bg-muted/20">
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-xl border border-border text-xs text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          >
            انصراف
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs shadow transition-colors"
          >
            تأیید و اعمال فیلتر
          </button>
        </div>
      </div>
    </div>
  )
}
