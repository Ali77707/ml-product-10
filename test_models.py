import warnings
warnings.filterwarnings('ignore')
from data import generate_dataset
from models import SentimentAnalyzer

df = generate_dataset(300)
a = SentimentAnalyzer(df)
before, after = a.train_all_models()
for name, m in after.items():
    print(f"{name}: F1={m['metrics']['f1']:.3f} params={m['best_params']}")
print(f"Best: {a.best_model_name}")
a.generate_report('student_test.png')
print("Report saved")
