# Multiview scene estimate. header.frame_id is the robot base frame and
# header.stamp is the stamp of the observation the estimate was computed from.
std_msgs/Header header

# Versioned board schema that defines the target IDs, e.g. "aic_v1".
string schema

TargetEstimate[] targets
