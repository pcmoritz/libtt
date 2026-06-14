.section .rodata
.balign 16
.global @DATA_SYMBOL@
@DATA_SYMBOL@:
  .incbin "@DATA_PATH@"
@DATA_SYMBOL@End:
.balign 8
.global @SIZE_SYMBOL@
@SIZE_SYMBOL@:
  .quad @DATA_SYMBOL@End - @DATA_SYMBOL@
.global @FINGERPRINT_SYMBOL@
@FINGERPRINT_SYMBOL@:
  .incbin "@FINGERPRINT_PATH@"
  .byte 0
.section .note.GNU-stack,"",@progbits
