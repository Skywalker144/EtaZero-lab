#pragma once
namespace etazero {
// Board storage, independent of the NN canvas and the action vocabulary.
// Only x<width, y<height are playable; padding cells are not board points.
struct BoardGeometry {
    int width,height,stride;
    int area() const { return width*height; }
    int storage_size() const { return stride*height; }
    int index(int x,int y) const { return y*stride+x; }
};
}
