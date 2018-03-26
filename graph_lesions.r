library('ggplot2')

df = read.csv('data/lesions.csv', header=FALSE,
              col.names=c('lesion', 'AUC'))

df$lesion = c('none', 'user all', 'user id', 'user other', 'word all', 'word ids', 'word other', 'neighbors', 'external', 'temporal')

df$group = c('none', 'user', 'user', 'user', 'word', 'word', 'word', 'word', 'other', 'other')

df$lesion = factor(df$lesion, levels=df$lesion)

none_score = df[df$lesion == 'none', 'AUC']



ggplot(df, aes(x=lesion, y=AUC, fill=group)) + geom_bar(stat='identity') +
  geom_hline(yintercept=none_score) +
  scale_x_discrete(limits = rev(levels(df$lesion))) +
  scale_fill_discrete(name='lesion type', breaks=c('none', 'user', 'word', 'other')) +
  xlab('lesioned features') +
  ylab('AUROC') +
  coord_flip(ylim=c(.82, .845)) +
  theme_minimal()

ggsave('doc/naaclhlt2018-latex/lesions.pdf', height=3, width=8)
