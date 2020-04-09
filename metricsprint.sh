python -c "import dill; import dim.plan.runtime_metrics as runtime_metrics; mm=dill.load(open('metrics.dill','rb')); print(mm.n_total); mm.quantify(); print(mm.format_table()); mm.print_conclusion_statii(); print(); print(mm); runtime_metrics.print_tex(mm);"
# ; mm.plot_passenger_comfort()"
