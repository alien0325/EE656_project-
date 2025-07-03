import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_expression_metrics_table():
    """Create a comprehensive table of expression metrics from test results."""
    
    # Load test results
    with open('real_test_results/test_results.json', 'r') as f:
        results = json.load(f)
    
    # Extract per-class metrics
    per_class_accuracy = results['per_class_accuracy']
    classification_report = results['classification_report']
    
    # Create data for the table
    expressions = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    
    for expression in per_class_accuracy.keys():
        expressions.append(expression.title())
        accuracies.append(per_class_accuracy[expression])
        precisions.append(classification_report[expression]['precision'])
        recalls.append(classification_report[expression]['recall'])
        f1_scores.append(classification_report[expression]['f1-score'])
        supports.append(int(classification_report[expression]['support']))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Expression': expressions,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores,
        'Support': supports
    })
    
    # Format percentages
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        df[col] = df[col].apply(lambda x: f"{x:.3f} ({x*100:.1f}%)")
    
    # Sort by F1-score (descending)
    df_sorted = df.copy()
    df_sorted['F1-Score_Num'] = [float(x.split()[0]) for x in df['F1-Score']]
    df_sorted = df_sorted.sort_values('F1-Score_Num', ascending=False)
    df_sorted = df_sorted.drop('F1-Score_Num', axis=1)
    
    # Print the table
    print("=" * 100)
    print("DICE-FER MODEL PERFORMANCE METRICS")
    print("=" * 100)
    print(f"Overall Accuracy: {results['overall_accuracy']:.3f} ({results['overall_accuracy']*100:.1f}%)")
    print(f"Total Test Samples: {results['test_samples']:,}")
    print("=" * 100)
    
    # Print formatted table
    print(f"{'Expression':<12} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
    print("-" * 100)
    
    for _, row in df_sorted.iterrows():
        print(f"{row['Expression']:<12} {row['Accuracy']:<15} {row['Precision']:<15} {row['Recall']:<15} {row['F1-Score']:<15} {row['Support']:<10}")
    
    print("-" * 100)
    
    # Add macro and weighted averages
    macro_avg = classification_report['macro avg']
    weighted_avg = classification_report['weighted avg']
    
    print(f"{'MACRO AVG':<12} {'-':<15} {macro_avg['precision']:.3f} ({macro_avg['precision']*100:.1f}%) {'-':<15} {macro_avg['recall']:.3f} ({macro_avg['recall']*100:.1f}%) {'-':<15} {macro_avg['f1-score']:.3f} ({macro_avg['f1-score']*100:.1f}%) {'-':<10}")
    print(f"{'WEIGHTED AVG':<12} {'-':<15} {weighted_avg['precision']:.3f} ({weighted_avg['precision']*100:.1f}%) {'-':<15} {weighted_avg['recall']:.3f} ({weighted_avg['recall']*100:.1f}%) {'-':<15} {weighted_avg['f1-score']:.3f} ({weighted_avg['f1-score']*100:.1f}%) {'-':<10}")
    
    print("=" * 100)
    
    # Create a visual table using matplotlib
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for visualization (remove percentage formatting)
    df_viz = df_sorted.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        df_viz[col] = [float(x.split()[0]) for x in df_viz[col]]
    
    # Create table
    table_data = []
    for _, row in df_viz.iterrows():
        table_data.append([
            row['Expression'],
            f"{row['Accuracy']:.3f}",
            f"{row['Precision']:.3f}",
            f"{row['Recall']:.3f}",
            f"{row['F1-Score']:.3f}",
            str(row['Support'])
        ])
    
    # Add header
    headers = ['Expression', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support']
    table_data.insert(0, headers)
    
    # Create table with colors
    table = ax.table(cellText=table_data[1:], colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows based on F1-score performance
    for i in range(1, len(table_data)):
        f1_score = float(table_data[i][4])
        if f1_score >= 0.8:
            color = '#C8E6C9'  # Light green for high performance
        elif f1_score >= 0.6:
            color = '#FFF9C4'  # Light yellow for medium performance
        else:
            color = '#FFCDD2'  # Light red for low performance
        
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('DICE-FER Model Performance Metrics by Expression\n' + 
              f'Overall Accuracy: {results["overall_accuracy"]:.3f} ({results["overall_accuracy"]*100:.1f}%)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save the table
    plt.tight_layout()
    plt.savefig('real_test_results/expression_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save as CSV for further analysis
    df_viz.to_csv('real_test_results/expression_metrics_table.csv', index=False)
    
    print(f"\nTable saved as:")
    print(f"- Image: real_test_results/expression_metrics_table.png")
    print(f"- CSV: real_test_results/expression_metrics_table.csv")
    
    return df_sorted

def create_performance_summary():
    """Create a summary of key performance insights."""
    
    with open('real_test_results/test_results.json', 'r') as f:
        results = json.load(f)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY & INSIGHTS")
    print("=" * 80)
    
    # Best performing expressions
    per_class_accuracy = results['per_class_accuracy']
    best_expression = max(per_class_accuracy, key=per_class_accuracy.get)
    worst_expression = min(per_class_accuracy, key=per_class_accuracy.get)
    
    print(f"🏆 Best Performing Expression: {best_expression.title()} ({per_class_accuracy[best_expression]:.1%})")
    print(f"⚠️  Most Challenging Expression: {worst_expression.title()} ({per_class_accuracy[worst_expression]:.1%})")
    
    # Performance categories
    high_perf = [exp for exp, acc in per_class_accuracy.items() if acc >= 0.8]
    medium_perf = [exp for exp, acc in per_class_accuracy.items() if 0.6 <= acc < 0.8]
    low_perf = [exp for exp, acc in per_class_accuracy.items() if acc < 0.6]
    
    print(f"\n📊 Performance Categories:")
    print(f"   High Performance (≥80%): {', '.join([exp.title() for exp in high_perf])}")
    print(f"   Medium Performance (60-80%): {', '.join([exp.title() for exp in medium_perf])}")
    print(f"   Low Performance (<60%): {', '.join([exp.title() for exp in low_perf])}")
    
    # Dataset balance
    classification_report = results['classification_report']
    supports = {exp: int(classification_report[exp]['support']) for exp in per_class_accuracy.keys()}
    total_samples = sum(supports.values())
    
    print(f"\n📈 Dataset Balance:")
    for exp, support in sorted(supports.items(), key=lambda x: x[1], reverse=True):
        percentage = (support / total_samples) * 100
        print(f"   {exp.title()}: {support:,} samples ({percentage:.1f}%)")
    
    # Recommendations
    print(f"\n💡 Recommendations for Improvement:")
    if low_perf:
        print(f"   • Focus on improving {', '.join([exp.title() for exp in low_perf])} recognition")
        print(f"   • Consider data augmentation for underrepresented expressions")
        print(f"   • Implement class-balanced training strategies")
    
    if len(high_perf) < 3:
        print(f"   • Overall model performance needs improvement")
        print(f"   • Consider ensemble methods or advanced architectures")
    
    print(f"   • Current overall accuracy: {results['overall_accuracy']:.1%}")
    print(f"   • Target for production: ≥85% overall accuracy")

if __name__ == "__main__":
    # Create the metrics table
    df = create_expression_metrics_table()
    
    # Create performance summary
    create_performance_summary() 