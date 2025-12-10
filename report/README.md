# Final Project Report

**Course:** ECSE 397/600 – Efficient Deep Learning  
**Project:** Energy-Aware Quantization and Mixed-Precision Learning  
**Team:** Salma Bhar, Wiam Skakri, Krupa Venkatesan

## 📁 Files

- `main.tex` - Main LaTeX source file
- `neurips_2024.sty` - NeurIPS-style formatting
- `Makefile` - Build automation
- `figures/` - Generated plots and visualizations

## 🔧 Compilation Options

### Option 1: Local Compilation (Recommended)

If you have LaTeX installed locally (MacTeX, TeX Live, MiKTeX):

```bash
# Simple compilation
make

# or manually:
pdflatex main.tex
pdflatex main.tex  # Run twice for references

# View the PDF
make view
```

### Option 2: Overleaf (Easiest)

1. Create a new project on [Overleaf](https://www.overleaf.com/)
2. Upload all files from this `report/` directory
3. Set `main.tex` as the main document
4. Compile automatically

### Option 3: VS Code with LaTeX Workshop

1. Install the "LaTeX Workshop" extension
2. Open `main.tex`
3. Press `Cmd+Alt+B` (Mac) or `Ctrl+Alt+B` (Windows/Linux) to build

## 📊 Report Structure

1. **Abstract** - Summary of the work and key findings
2. **Introduction** - Motivation, problem statement, contributions
3. **Related Work** - PTQ, mixed-precision, energy-aware methods
4. **Methodology** - Quantization scheme, sensitivity analysis, energy model
5. **Experiments** - Results on ResNet-18 and DeiT-Tiny
6. **Discussion** - Analysis, CNN vs Transformer robustness, limitations
7. **Conclusion** - Summary and future work
8. **References** - Properly formatted citations

## 📈 Key Results

| Model | Config | Accuracy | Energy Savings |
|-------|--------|----------|----------------|
| ResNet-18 | FP32 | 61.44% | — |
| ResNet-18 | Mixed | 60.80% | 85.7% |
| DeiT-Tiny | FP32 | 85.34% | — |
| DeiT-Tiny | Mixed | 85.05% | 80.9% |

## 🔍 Editing Tips

- Update accuracy numbers in Tables 1 and 2 if you re-run experiments
- Add more figures by placing them in `figures/` and referencing in LaTeX
- The `\rowcolor{pareto!20}` highlights the mixed-precision rows
- Limitations and future work are in Section 5

## 📝 Checklist Before Submission

- [ ] All accuracy numbers match latest experimental results
- [ ] Author names and emails are correct
- [ ] Figures are properly labeled and referenced
- [ ] References are complete and properly formatted
- [ ] No placeholder text remains
- [ ] PDF compiles without errors

