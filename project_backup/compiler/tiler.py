class Tiler:

    def __init__(self, tile_size=64):

        self.tile_size = tile_size

    def tile_matrix(self, rows, cols):

        row_tiles = []

        for r in range(0, rows, self.tile_size):

            for c in range(0, cols, self.tile_size):

                row_tiles.append(
                    (r, c)
                )

        return row_tiles
    

