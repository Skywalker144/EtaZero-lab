// Built only by tests/reference/check_katagomo_rules.py against the fixed external source.
#include "ForbiddenPointFinder.h"
#include "etazero/rules.h"
#include <iostream>
#include <random>
#include <map>
#include <cmath>
using etazero::Forbidden;
size_t checked=0;
std::map<int,size_t> categories;
void compare(const etazero::Board& b,int x,int y) {
    CForbiddenPointFinder ref(b.size);
    for(int yy=0;yy<b.size;++yy)for(int xx=0;xx<b.size;++xx)
        ref.SetStone(xx,yy,b.cells[yy*b.size+xx]==-1?C_WHITE:b.cells[yy*b.size+xx]);
    // Authoritative recursive implementation, including its nearby early-out.
    bool expected=ref.isForbidden(x,y);
    etazero::RenjuAnalyzer analyzer;
    auto category=analyzer.analyze(b,y*b.size+x);
    if(expected!=(category!=Forbidden::NONE))throw std::runtime_error("Forbidden mismatch at case "+std::to_string(checked));
    auto expected_category=ref.IsFive(x,y,C_BLACK)?Forbidden::NONE:
        ref.IsOverline(x,y)?Forbidden::OVERLINE:
        ref.IsDoubleFour(x,y)?Forbidden::DOUBLE_FOUR:
        ref.IsDoubleThree(x,y)?Forbidden::DOUBLE_THREE:Forbidden::NONE;
    if(category!=expected_category)throw std::runtime_error("Forbidden category mismatch at case "+std::to_string(checked));
    ++checked;++categories[static_cast<int>(category)];
}
void points(int size,int x,int y,std::initializer_list<std::pair<int,int>> black,Forbidden expected) {
    etazero::Board b(size);
    for(auto [dx,dy]:black)b.cells[(y+dy)*size+x+dx]=1;
    etazero::RenjuAnalyzer a;
    if(a.analyze(b,y*size+x)!=expected)throw std::runtime_error("Named boundary case "+std::to_string(checked)+" expected "+std::to_string(static_cast<int>(expected))+" got "+std::to_string(static_cast<int>(a.analyze(b,y*size+x))));
    compare(b,x,y);
}
int main() {
    try {
        std::mt19937_64 rng(309);
        for(int size:{11,12,13,14,15}) {
            int c=size/2;
            // Exact five beats an overline in another direction, as in IsOverline.
            points(size,c,c,{{-2,0},{-1,0},{1,0},{2,0},{0,-3},{0,-2},{0,-1},{0,1},{0,2}},Forbidden::NONE);
            points(size,c,c,{{-3,0},{-2,0},{-1,0},{1,0},{2,0}},Forbidden::OVERLINE);
            points(size,c,c,{{-1,0},{1,0},{2,0},{0,-1},{0,1},{0,2}},Forbidden::DOUBLE_FOUR);
            // Two distinct broken fours in the same direction.
            points(size,c,c,{{-3,0},{-1,0},{1,0},{3,0}},Forbidden::DOUBLE_FOUR);
            points(size,c,c,{{-1,0},{1,0},{0,-1},{0,1}},Forbidden::DOUBLE_THREE);
            points(size,1,1,{{-1,0},{1,0},{0,-1},{0,1}},Forbidden::NONE);
            // Each extension of the apparent horizontal three makes a vertical
            // overline: a pseudo-three cannot contribute to double-three.
            points(size,c,c,{{-1,0},{1,0},{0,-1},{0,1},{-2,-3},{-2,-2},{-2,-1},{-2,1},{-2,2},{2,-3},{2,-2},{2,-1},{2,1},{2,2}},Forbidden::NONE);
            // Exhaust all 3^8 immediate-neighbour states, including White blockers.
            for(int pattern=0;pattern<6561;++pattern)for(int edge=0;edge<2;++edge) {
                int x=edge?1:c,y=edge?1:c,p=pattern;
                etazero::Board b(size);
                for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)if(dx||dy) {
                    int stone=p%3;p/=3;b.cells[(y+dy)*size+x+dx]=stone==2?-1:stone;
                }
                compare(b,x,y);
            }
            // Exhaust ten cells on each axis, with an independent perpendicular
            // pattern. Cross-axis extensions exercise recursive live-three checks.
            for(int d=0;d<4;++d)for(int mask=0;mask<1024;++mask) {
                etazero::Board b(size);
                for(int offset=-5;offset<=5;++offset)if(offset) {
                    int bit=offset<0?offset+5:offset+4;
                    int loc=b.offset(c*size+c,d,offset);
                    if(loc>=0)b.cells[loc]=(mask>>bit)&1;
                }
                for(int offset=-4;offset<=4;++offset)if(offset) {
                    int loc=b.offset(c*size+c,(d+1)%4,offset);
                    auto p=rng()%5;if(loc>=0)b.cells[loc]=p<2?1:p==2?-1:0;
                }
                compare(b,c,c);
            }
            // Dense/sparse local patterns, corners and all four board boundaries.
            for(int test=0;test<10000;++test) {
                int x=test%3==0?0:test%3==1?size-1:c,y=test%4==0?0:test%4==1?size-1:c;
                etazero::Board b(size);
                for(int yy=0;yy<size;++yy)for(int xx=0;xx<size;++xx)
                    if(std::abs(xx-x)<=5 && std::abs(yy-y)<=5 && (xx!=x||yy!=y)) {
                        int p=rng()%100;b.cells[yy*size+xx]=p<35?1:p<45?-1:0;
                    }
                compare(b,x,y);
            }
        }
        for(auto category:{Forbidden::NONE,Forbidden::OVERLINE,Forbidden::DOUBLE_FOUR,Forbidden::DOUBLE_THREE})
            if(!categories[static_cast<int>(category)])throw std::runtime_error("Missing outcome coverage");
        std::cout<<"{\"cases\":"<<checked<<",\"seed\":309,\"sizes\":[11,12,13,14,15],\"categories\":{\"none\":"<<categories[0]
            <<",\"overline\":"<<categories[1]<<",\"double_four\":"<<categories[2]<<",\"double_three\":"<<categories[3]<<"}}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
