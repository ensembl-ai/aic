#
#  Copyright (C) 2026 Ensemble Robotics
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import os

# External

from launch import LaunchContext
from launch_ros.substitutions import FindPackageShare
from tesseract_robotics.tesseract_common import (
    FilesystemPath,
    GeneralResourceLocator,
)
from tesseract_robotics.tesseract_environment import Environment
import xacro


class EnsemblRobot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.locator = self._create_resource_locator()
        self.env = Environment()
        context = LaunchContext()
        self.aic_description_share = FindPackageShare("aic_description").perform(context)
        self.urdf_xacro_path = f"{self.aic_description_share}/urdf/ur_gz.urdf.xacro"
        self.srdf_path = f"{self.aic_description_share}/srdf/ur_gz.srdf"
        urdf_xml = self._expand_xacro(self.urdf_xacro_path)
        srdf_xml = self._read_srdf(self.srdf_path)
        ok = self.env.init(
            urdf_xml,
            srdf_xml,
            self.locator,
        )
        if not ok:
            raise RuntimeError("Failed to initialize tesseract environment")
        else:
            self.logger.info("Tesseract environment initialized successfully")

    def _create_resource_locator(self) -> GeneralResourceLocator:
        locator = GeneralResourceLocator()
        for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
            if not prefix:
                continue
            share_dir = os.path.join(prefix, "share")
            if os.path.isdir(share_dir):
                locator.addPath(FilesystemPath(share_dir))
        return locator

    def _expand_xacro(self, xacro_path: str) -> str:
        doc = xacro.process_file(xacro_path, mappings={"name": "ur"})
        return doc.toxml()

    def _read_srdf(self, srdf_path: str) -> str:
        with open(srdf_path, encoding="utf-8") as f:
            return f.read()


if __name__ == "__main__":
    robot = EnsemblRobot()
