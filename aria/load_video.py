# Copyright 2024 Rhymes AI. All rights reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_video(video_file, num_frames=8):
    import decord
    from decord import VideoReader

    decord.bridge.set_bridge("torch")

    vr = VideoReader(video_file)
    duration = len(vr)

    frame_indices = [int(duration / num_frames) * i for i in range(num_frames)]
    frames = vr.get_batch(frame_indices).numpy()
    return [Image.fromarray(fr).convert("RGB") for fr in frames]
