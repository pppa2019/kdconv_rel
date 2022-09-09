# rebuild code target
change each model to a single py file -- done
change each solver to a single py file -- done
join solvers and models -- done
simplifiy the dataloading process and least duplication
change model saving dir to a individual folder -- done

# current progress
拆分了代码的结构，当前发现先前并不使用的带有应变量的模型在加入pointer的decoder上并不能很好地适配，主要是输入的sentence embedding 为全部对话而非遮盖了golden的。该代码需要check一下，为什么需要加入sentence_num+batch_size的句子。
