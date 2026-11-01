# Estimate for one named target of the board schema, such as
# "task_board", "nic_card_mount_3/sfp_port_1", or "sc_port_0".
string id

# Target category: "board", "sfp_port", or "sc_port".
string type

# True when presence_probability exceeds the configured threshold.
bool present
float32 presence_probability

# Predicted probability that the pose error is within the trained tolerance.
float32 confidence

# Target pose in the SceneEstimate header frame.
geometry_msgs/Pose pose
