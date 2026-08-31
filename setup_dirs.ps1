$dirs = @(
  "src\extraction", "src\preprocessing", "src\chunking", "src\embedding",
  "src\vectorstore", "src\retrieval", "src\generation", "src\evaluation",
  "src\experiment", "src\analysis",
  "scripts", "tests\fixtures", "notebooks\exploratory", "config", "questions", "docs",
  "data\raw", "data\extracted", "data\preprocessed",
  "data\chunks\dev\fixed_char", "data\chunks\dev\recursive_char", "data\chunks\dev\paragraph", "data\chunks\dev\heading", "data\chunks\dev\eluc",
  "data\chunks\formal\fixed_char", "data\chunks\formal\recursive_char", "data\chunks\formal\paragraph", "data\chunks\formal\heading", "data\chunks\formal\eluc",
  "results\dev", "results\runs", "results\analysis", "results\figures",
  "chromadb\dev", "chromadb\experiment", "app\components"
)

New-Item -ItemType File -Force -Path "src\__init__.py" | Out-Null

foreach ($dir in $dirs) {
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  New-Item -ItemType File -Force -Path "$dir\.gitkeep" | Out-Null
  if ($dir -like "src\*") {
    New-Item -ItemType File -Force -Path "$dir\__init__.py" | Out-Null
  }
}
