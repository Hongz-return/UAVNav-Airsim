"""UAV AirSim 实时 RViz 监控台（纯在线、严格只读）。

启动::

    pixi run python -m tools.uav_rviz --port 42072 --mode bypass --open

模块职责
--------
- ``airsim_tap``      只读 AirSim 采集器（坐标系转换、针孔反投影、双模式限速与退避）。
- ``geometry_loader`` 只读障碍物 / 目标几何加载（simListSceneObjects + course_episodes.jsonl）。
- ``protocol``        紧凑二进制遥测帧编解码（Header + 点云 XYZ Float32 + 深度 uint8/uint16）。
- ``ws_server``       asyncio 静态托管 + RFC6455 WebSocket 服务端 + 控制指令路由。
- ``simulated``       离线模拟 AirSim 数据源（``--demo``，用于无仿真器时验证前端）。

本包只依赖 Python 标准库、numpy 以及已安装的 cosysairsim，不新增任何第三方依赖。
"""

__version__ = "1.0.0"
