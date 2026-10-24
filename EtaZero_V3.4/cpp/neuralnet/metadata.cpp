#include "neuralnet/metadata.h"
#include "neuralnet/inference.h"
#include <cmath>
#include <numeric>
namespace etazero {
SGFMetadata selfplay_metadata(const std::string& date) {
    SGFMetadata metadata;
    metadata.initialized=true;
    metadata.bRankIsUnknown=metadata.wRankIsUnknown=true;
    metadata.gameIsUnrated=true;
    metadata.tcIsNone=true;
    metadata.gameDate=SimpleDate(date);
    return metadata;
}
bool Backend::available() const {return !metadata_channels() || (metadata_profile_ && metadata_profile_->initialized);}
void Backend::validate_human_profile(const std::string& name) const {
    if(!metadata_channels())throw std::runtime_error("Model has no metadata encoder");
    if(name!="selfplay")SGFMetadata::getProfile(name);
}
void Backend::set_human_profile(const std::string& name) {
    validate_human_profile(name);
    selfplay_metadata_=name=="selfplay";
    metadata_profile_=std::make_shared<SGFMetadata>(selfplay_metadata_?selfplay_metadata(DateTime::getDateString()):SGFMetadata::getProfile(name));
}
void Backend::append_metadata(std::vector<float>& input,int player) const {
    if(!metadata_channels())return;
    if(!available())throw std::runtime_error("Metadata profile is not set");
    append_profile(input,player,*metadata_profile_);
}
void Backend::append_profile(std::vector<float>& input,int player,const SGFMetadata& metadata) const {
    if(metadata_channels()!=SGFMetadata::METADATA_INPUT_NUM_CHANNELS || !metadata.initialized)
        throw std::runtime_error("Missing metadata encoder or uninitialized metadata");
    if(player!=1 && player!=-1)throw std::runtime_error("Metadata requires the actual side to move");
    const size_t area=(input.size()-spec_.globals)/spec_.planes;
    const int board_area=std::lround(std::accumulate(input.begin(),input.begin()+area,0.0f));
    size_t start=input.size();input.resize(start+SGFMetadata::METADATA_INPUT_NUM_CHANNELS);
    SGFMetadata::fillMetadataRow(&metadata,input.data()+start,player==1?P_BLACK:P_WHITE,board_area);
}
}
