#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Stage 9 ResponseSynthesizer structured summary parsing
and incremental caching (Feature 006: User Story 5).
"""

import pytest
from unittest.mock import MagicMock

from rag_core.generation.synthesizer import ResponseSynthesizer, SynthesisResult


def test_parse_answer_and_summary_html_comment_tag():
    """Verify that <!-- SUMMARY --> tags are extracted into summary and stripped from answer."""
    raw_completion = (
        "چگالی برابر با نسبت جرم به حجم است.\n\n"
        "<!-- SUMMARY -->\n"
        "بررسی مفهوم چگالی و رابطه ریاضی ρ = m / V در فصل دوم علوم هفتم.\n"
        "<!-- /SUMMARY -->"
    )

    clean_answer, summary = ResponseSynthesizer.parse_answer_and_summary(raw_completion)

    assert "<!-- SUMMARY -->" not in clean_answer
    assert "<!-- /SUMMARY -->" not in clean_answer
    assert clean_answer.strip() == "چگالی برابر با نسبت جرم به حجم است."
    assert summary == "بررسی مفهوم چگالی و رابطه ریاضی ρ = m / V در فصل دوم علوم هفتم."


def test_parse_answer_and_summary_with_preamble_and_notes():
    """Verify that preamble instructions and cautionary notes are stripped from answer and summary."""
    raw_completion = (
        "چگالی برابر با نسبت جرم به حجم است.\n\n"
        "دستورالعمل ثبت خلاصه در حافظه گفت‌وگو:\n"
        "<!-- SUMMARY -->\n"
        "بررسی مفهوم چگالی و رابطه ریاضی ρ = m / V در فصل دوم علوم هفتم.\n"
        "<!-- /SUMMARY -->\n"
        "توجه بسیار مهم: از خلاصه‌سازی پیام‌ها یا نوبت‌های پیشین خودداری کنید."
    )

    clean_answer, summary = ResponseSynthesizer.parse_answer_and_summary(raw_completion)

    assert "دستورالعمل" not in clean_answer
    assert "توجه بسیار مهم" not in clean_answer
    assert "<!-- SUMMARY -->" not in clean_answer
    assert clean_answer.strip() == "چگالی برابر با نسبت جرم به حجم است."
    assert summary == "بررسی مفهوم چگالی و رابطه ریاضی ρ = m / V در فصل دوم علوم هفتم."


def test_parse_answer_and_summary_markdown_header():
    """Verify that Persian markdown headers are extracted as fallback summary marker."""
    raw_completion = (
        "سلول واحد ساختار موجودات زنده است.\n\n"
        "### خلاصه پرسش و پاسخ:\n"
        "یاخته‌ها دارای غشا، هسته و سیتوپلاسم هستند."
    )

    clean_answer, summary = ResponseSynthesizer.parse_answer_and_summary(raw_completion)

    assert clean_answer.strip() == "سلول واحد ساختار موجودات زنده است."
    assert summary == "یاخته‌ها دارای غشا، هسته و سیتوپلاسم هستند."


def test_parse_answer_and_summary_no_tags():
    """When no tags exist, raw text is returned and summary is None."""
    raw_completion = "این یک پاسخ بدون برچسب خلاصه است."
    clean_answer, summary = ResponseSynthesizer.parse_answer_and_summary(raw_completion)

    assert clean_answer == raw_completion
    assert summary is None


def test_synthesizer_populates_q_and_a_summary_from_llm():
    """
    ResponseSynthesizer.synthesize should parse LLM completion,
    strip tags from answer, and populate q_and_a_summary.
    """
    synth = ResponseSynthesizer(base_url="http://localhost:20128/v1")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "دستگاه گردش خون شامل قلب و رگ‌هاست.\n\n"
        "<!-- SUMMARY -->\n"
        "گردش خون و وظایف سرخرگ‌ها و سیاهرگ‌ها.\n"
        "<!-- /SUMMARY -->"
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 30
    mock_response.usage.total_tokens = 80
    mock_client.chat.completions.create.return_value = mock_response

    synth._client = mock_client

    sample_chunk = {
        "text": "قلب خون را به تمام بدن پمپاژ می‌کند.",
        "page": 10,
        "doc_id": "science_textbook.pdf"
    }

    result = synth.synthesize(query="قلب چگونه کار می‌کند؟", chunks=[sample_chunk])

    assert isinstance(result, SynthesisResult)
    assert "<!-- SUMMARY -->" not in result.answer
    assert "دستگاه گردش خون" in result.answer
    assert result.q_and_a_summary == "گردش خون و وظایف سرخرگ‌ها و سیاهرگ‌ها."


def test_synthesizer_fallback_summary_when_tag_missing():
    """
    When LLM output does not include a summary tag,
    q_and_a_summary defaults to the first 250 characters of the answer.
    """
    synth = ResponseSynthesizer(base_url="http://localhost:20128/v1")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "انرژی پتانسیل گرانشی به ارتفاع و جرم جسم بستگی دارد."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 40
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 60
    mock_client.chat.completions.create.return_value = mock_response

    synth._client = mock_client

    sample_chunk = {
        "text": "فرمول انرژی پتانسیل U = mgh است.",
        "page": 25,
        "doc_id": "science_textbook.pdf"
    }

    result = synth.synthesize(query="عوامل موثر بر انرژی پتانسیل چیست؟", chunks=[sample_chunk])

    assert isinstance(result, SynthesisResult)
    assert result.q_and_a_summary == "انرژی پتانسیل گرانشی به ارتفاع و جرم جسم بستگی دارد."


def test_synthesizer_does_not_duplicate_history_when_history_chunks_present():
    """
    When chunks already contains conversation history (Decision Model 2 managed),
    synthesizer must NOT append raw history turns into messages to prevent double injection.
    """
    synth = ResponseSynthesizer(base_url="http://localhost:20128/v1")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "دهان اولین بخش لوله گوارش است."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    synth._client = mock_client

    history_chunk = {
        "text": "خلاصه گفتگوی پیشین: بررسی دستگاه گوارش",
        "page": 1,
        "breadcrumb": "گفتگوی پیشین",
        "doc_id": "conversation_history",
        "chunk_id": "turn_1"
    }

    raw_history = [
        {"role": "user", "content": "دستگاه گوارش چیست؟"},
        {"role": "assistant", "content": "دستگاه گوارش شامل دهان و معده است."}
    ]

    synth.synthesize(
        query="نقش بخش اول چیست؟",
        chunks=[history_chunk],
        history=raw_history
    )

    call_args = mock_client.chat.completions.create.call_args[1]
    messages = call_args["messages"]

    # Only system and the final user message should be in messages (no raw history duplicates)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "خلاصه گفتگوی پیشین: بررسی دستگاه گوارش" in messages[1]["content"]

