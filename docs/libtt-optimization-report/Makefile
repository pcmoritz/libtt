PANDOC ?= pandoc
REPORT := report

.PHONY: all html tex pdf pdf-html clean

all: html

html: $(REPORT).html

$(REPORT).html: $(REPORT).md style.css figures/throughput.svg figures/incremental-speedup.svg figures/upstream-comparison.svg figures/stack.svg
	$(PANDOC) $(REPORT).md \
		--from=markdown+tex_math_dollars \
		--to=html5 --standalone --toc --number-sections --mathml \
		--self-contained --css=style.css \
		--metadata pagetitle="Accelerating Qwen3-8B Decode on Tenstorrent Blackhole" \
		-o $@
	sed -i 's/[[:space:]]\+$$//' $@

tex: $(REPORT).tex

$(REPORT).tex: $(REPORT).md figures/throughput.svg figures/incremental-speedup.svg figures/upstream-comparison.svg figures/stack.svg
	$(PANDOC) $(REPORT).md \
		--from=markdown+tex_math_dollars \
		--to=latex --standalone --toc --number-sections \
		-o $@

pdf: $(REPORT).pdf

$(REPORT).pdf: $(REPORT).md figures/throughput.svg figures/incremental-speedup.svg figures/upstream-comparison.svg figures/stack.svg
	@command -v xelatex >/dev/null || (echo "xelatex is required (for example, texlive-xetex)"; exit 1)
	@command -v rsvg-convert >/dev/null || (echo "rsvg-convert is required (for example, librsvg2-bin)"; exit 1)
	$(PANDOC) $(REPORT).md \
		--from=markdown+tex_math_dollars \
		--pdf-engine=xelatex --toc --number-sections \
		-o $@

pdf-html: $(REPORT).html
	weasyprint $(REPORT).html $(REPORT).pdf

clean:
	rm -f $(REPORT).html $(REPORT).tex $(REPORT).pdf
