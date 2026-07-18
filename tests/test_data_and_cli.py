import pytest

from adaptivelearning.cli import main
from adaptivelearning.data import DataError, load_series


@pytest.fixture
def csv_file(tmp_path):
    def make(content, name="data.csv"):
        path = tmp_path / name
        path.write_text(content)
        return str(path)

    return make


def test_plain_numbers(csv_file):
    path = csv_file("1\n2\n3.5\n")
    assert load_series(path) == [1.0, 2.0, 3.5]


def test_csv_with_header_picks_last_numeric_column(csv_file):
    path = csv_file("date,label,close\n2024-01-01,a,101.5\n2024-01-02,b,102.25\n")
    assert load_series(path) == [101.5, 102.25]


def test_csv_column_by_name_case_insensitive(csv_file):
    path = csv_file("Open,Close\n10,11\n12,13\n")
    assert load_series(path, column="open") == [10.0, 12.0]


def test_csv_column_by_index(csv_file):
    path = csv_file("5,100\n6,200\n")
    assert load_series(path, column="0") == [5.0, 6.0]


def test_dollar_signs_and_thousands_separators(csv_file):
    path = csv_file('price\n"$1,234.50"\n"$1,240.00"\n')
    assert load_series(path) == [1234.5, 1240.0]


def test_missing_column_lists_available(csv_file):
    path = csv_file("open,close\n1,2\n")
    with pytest.raises(DataError, match="available: open, close"):
        load_series(path, column="volume")


def test_non_numeric_cell_raises_with_line_number(csv_file):
    path = csv_file("1\n2\noops\n")
    with pytest.raises(DataError, match="line 3"):
        load_series(path)


def test_empty_file_raises(csv_file):
    with pytest.raises(DataError):
        load_series(csv_file(""))


def test_blank_lines_skipped(csv_file):
    path = csv_file("1\n\n2\n\n")
    assert load_series(path) == [1.0, 2.0]


# ------------------------------------------------------------------ CLI


def test_cli_predict(csv_file, capsys):
    path = csv_file("\n".join(str(2 * t) for t in range(30)))
    assert main(["predict", path]) == 0
    out = capsys.readouterr().out
    assert "Predicted next value:" in out
    # A clean ramp: prediction should be close to 60.
    predicted = float(out.split("Predicted next value:")[1].split()[0])
    assert predicted == pytest.approx(60.0, abs=1.0)


def test_cli_backtest(csv_file, capsys):
    path = csv_file("\n".join(str(t + (t % 3)) for t in range(50)))
    assert main(["backtest", path]) == 0
    out = capsys.readouterr().out
    assert "ENSEMBLE" in out
    assert "Final weights" in out


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Final leaderboard" in out
    assert "Predicted next value:" in out


def test_cli_predict_too_few_points(csv_file, capsys):
    path = csv_file("1\n2\n")
    assert main(["predict", path]) == 1


def test_cli_missing_file(capsys):
    assert main(["predict", "/nonexistent/file.csv"]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_data_error_is_reported(csv_file, capsys):
    path = csv_file("a,b\nx,y\n")
    assert main(["predict", path]) == 1
    assert "error:" in capsys.readouterr().err
