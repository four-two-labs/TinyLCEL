#!/usr/bin/env python3
"""
Example: Extract page content as Markdown from a PDF using TinyLCEL message chunks.
Usage:
    python examples/extract_pages_markdown.py <pdf_path> [--first-page N] [--last-page M] [--dpi D]
"""
import argparse
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
import pdf2image  # type: ignore
from PIL import Image  # type: ignore

from tinylcel.runnable import Runnable
from tinylcel.messages import SystemMessage
from tinylcel.runnable import RunnableLambda
from tinylcel.messages.chunks import TextChunk
from tinylcel.chat_models.openai import ChatOpenAI
from tinylcel.output_parsers import StrOutputParser
from tinylcel.messages.image_chunk import ImageChunk
from tinylcel.chat_models.openai import BaseChatModel


def create_extract_chain(llm: BaseChatModel) -> Runnable[Image.Image, str]:
    """Build a TinyLCEL chain that extracts page content as Markdown."""
    return (
        RunnableLambda(
            lambda img: [
                TextChunk(
                    "You are a helpful assistant that converts the content of an image " 
                    "into Markdown, preserving the original layout and structure as much "
                    "as possible."
                )
                + TextChunk('##IMAGE##')
                + ImageChunk(img)
            ]
        )
        | llm
        | StrOutputParser()
    )


def extract_pages_markdown(
    llm: BaseChatModel,
    pdf_path: Union[str, Path],    
    first_page: int = 1,
    last_page: int = 10,
    dpi: int = 70,
) -> list[str]:
    """
    Returns a list of Markdown strings, one per page.
    """
    chain = create_extract_chain(llm)
    images = pdf2image.convert_from_path(
        str(pdf_path), 
        fmt="png", 
        dpi=dpi,
        first_page=first_page, 
        last_page=last_page
    )

    markdown_pages: list[str] = []
    for idx, img in enumerate(images, start=first_page):
        try:
            title = f'Page {idx:02d}'            
            page_md = chain.invoke(img)
            markdown_pages.append(page_md)
            
            print(f'{title:-^50}')
            print(page_md)            
            print()
        except Exception as e:
            print(f'Error processing page {idx}: {e}')
            
    return markdown_pages


def main():
    parser = argparse.ArgumentParser(
        description="Extract PDF pages as Markdown using TinyLCEL."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file.")
    parser.add_argument("--first-page", type=int, default=1, help="First page to process.")
    parser.add_argument("--last-page", type=int, default=10, help="Last page to process.")
    parser.add_argument("--dpi", type=int, default=70, help="DPI for PDF->image conversion.")
    args = parser.parse_args()

    load_dotenv()
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.0,
        max_completion_tokens=2048,
        max_retries=3,
        timeout=120,
    )

    extract_pages_markdown(
        llm=llm,
        pdf_path=args.pdf_path,
        first_page=args.first_page,
        last_page=args.last_page,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main() 