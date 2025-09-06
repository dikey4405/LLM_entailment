from data_loader import load_generated_answers, load_knowledge
from entailment_checker import EntailmentChecker

# Đường dẫn file
generated_file = ".../submit.csv"
knowledge_file = ".../vihallu-warmup.csv"

# Load dữ liệu
generated_answers = load_generated_answers(generated_file)
knowledge = load_knowledge(knowledge_file)

# Tạo đối tượng checker
checker = EntailmentChecker(threshold=0.75)

# Kiểm tra entailment
df_results = checker.check_entailment_loop(knowledge, generated_answers, output_file="entailment_results.csv")

print(df_results.head())

# Lọc các dòng entailed
entailed_rows = df_results[df_results['Entailed'] == True]
print(entailed_rows.head())

