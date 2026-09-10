module tpu (
    // AXI-Lite slave (control)
    axi_lite_if ctrl,

    // AXI4 master (DMA)
    axi_if dma
);

endmodule
