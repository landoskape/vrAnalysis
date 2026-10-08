from dimensionality_manuscript.scripts.status import print_error_summary


class _Store:
    def get_errors(self):
        return [
            {
                "session_id": "mouse.2020-01-01.1",
                "analysis_type": "analysis",
                "analysis_summary": "config",
                "schema_version": "v1",
                "error_message": "",
                "traceback": "Traceback (most recent call last):\n  ...\nEOFError\n",
            },
            {
                "session_id": "mouse.2020-01-02.1",
                "analysis_type": "analysis",
                "analysis_summary": "config",
                "schema_version": "v1",
                "error_message": "ordinary failure\nwith details",
                "traceback": "",
            },
            {
                "session_id": "mouse.2020-01-03.1",
                "analysis_type": "analysis",
                "analysis_summary": "config",
                "schema_version": "v1",
                "error_message": None,
                "traceback": None,
            },
        ]


def test_error_type_summary_handles_empty_exception_messages(capsys):
    print_error_summary(_Store(), include_error_types=True)

    output = capsys.readouterr().out
    assert "1x  EOFError" in output
    assert "1x  ordinary failure" in output
    assert "1x  <no error message>" in output
