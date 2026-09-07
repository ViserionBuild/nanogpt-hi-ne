zip -r nepali_pipeline_kaggle.zip . \
  -x "*/data/*" \
     "*/data/**" \
     ".git/*" \
     ".venv/*" \
     "*/__pycache__/*" \
     "*.pyc"


Order

So the simple answer is:

collect_news → collect_wikipedia → 01d_aggregate

And internally:

collect_wikipedia → 01b → 01c

zip -r hindi_for_mac.zip . \
  -x "*/data/*" "*/data/**" ".git/*" ".venv/*" "*/__pycache__/*" "*.pyc"

zip -r hindi_for_mac.zip hindi/data/intermediate/quality/

zip -r hindi_for_mac.zip . \
  -x "nepali/*" "hindi/data/raw/*" ".git/*" ".venv/*" "*/__pycache__/*" "*.pyc"
