"""Reference previews are local evaluation inputs, never model context."""

from pathlib import Path

from src.evaluation import parse_reference
from src.reference_tables import (
    parsed_table_reference, read_reference_tables, recommend_columns,
    suggest_header_row, suggest_table, table_columns,
)
from server.models import ReferenceOptions

MAX_REFERENCE_BYTES = 1024 * 1024


def preview_reference(data: bytes, filename: str, options: ReferenceOptions) -> dict:
    if len(data) > MAX_REFERENCE_BYTES:
        raise ValueError("The reference must be 1 MiB or smaller.")
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in {".txt", ".srt", ".vtt", ".xlsx", ".csv", ".tsv"}:
        raise ValueError("Choose a TXT, SRT, VTT, XLSX, CSV, or TSV reference.")
    result = {"tables": [], "sheet": None, "header_row": None,
              "source_column": None, "english_column": None, "needs_source_selection": False}
    if suffix not in {".xlsx", ".csv", ".tsv"}:
        reference = dict(parse_reference(data, name, format=options.format), name=name)
        return dict(result, reference=reference, reference_view=dict(reference, title="Reference transcript"))

    tables = read_reference_tables(data, name)
    sheet = options.sheet or suggest_table(tables)
    if sheet not in tables:
        raise ValueError("Choose a worksheet that exists in this reference.")
    for table_name, rows in tables.items():
        header_number = (options.header_row if table_name == sheet and options.header_row is not None
                         else suggest_header_row(rows) + 1)
        header = header_number - 1 if header_number else None
        suggested = recommend_columns(rows, header)
        result["tables"].append({
            "name": table_name, "columns": table_columns(rows, header),
            "suggested_source": suggested["source"], "suggested_english": suggested["display"],
            "header_row": header_number, "row_count": len(rows),
        })
    selected = next(table for table in result["tables"] if table["name"] == sheet)
    source_column = (options.source_column if "source_column" in options.model_fields_set
                     else selected["suggested_source"])
    english_column = (options.english_column if "english_column" in options.model_fields_set
                      else selected["suggested_english"])
    result.update(sheet=sheet, header_row=selected["header_row"], source_column=source_column,
                  english_column=english_column, needs_source_selection=source_column is None)
    if source_column is None:
        return dict(result, reference=None, reference_view=None)

    def parse_column(column):
        header = selected["header_row"] - 1 if selected["header_row"] else None
        parsed = parsed_table_reference(tables[sheet], header_row=header,
                                        text_column=column, format=options.format)
        return dict(parsed, name=name, sheet=sheet, column=selected["columns"][column],
                    header_row=selected["header_row"])

    reference = parse_column(source_column)
    view = (dict(parse_column(english_column), title="English reference")
            if english_column is not None and english_column != source_column
            else dict(reference, title="Reference transcript"))
    return dict(result, reference=reference, reference_view=view)
