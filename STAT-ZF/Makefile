PARAMS=-N --pdf-engine=xelatex -V documentclass=scrartcl -V papersize=a4 -V lang=de -V urlcolor=blue --toc
FONTS=-V mainfont="IBM Plex Serif" -V sansfont="IBM Plex Sans" -V monofont="IBM Plex Mono"
DATE=-V date="`date +'%d.%m.%Y'`"

STAT_ZF.pdf: stat-zf.md
	pandoc -s $(PARAMS) $(FONTS) $(DATE) stat-zf.md -o STAT_ZF.pdf
