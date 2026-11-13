# FRAME batch image: Pinocchio from conda-forge, then frame-core, frame-kinematics and frame-batch.
#
# Build and run it through the ./frame wrapper (`./frame build`, `./frame run`, ...). The project's
# utils/ package (ID_Xsens + the MVNX converter), models/, batch.json, the dataset and the output
# folders are bind-mounted at run time, not copied in -- editing them needs no rebuild.
FROM condaforge/miniforge3:24.3.0-0

# Robotics stack from conda-forge (the reliable route for Pinocchio and its eigenpy bindings).
RUN mamba install -y -c conda-forge \
        python=3.11 pinocchio numpy scipy pandas matplotlib pyyaml \
    && mamba clean -a -y

# Runtime deps of utils/ (imported by ID_Xsens and the URDF builder):
#   pin-pink                     -> differential IK in ID_Xsens.runFullIK
#   qpsolvers + quadprog         -> pink's QP backend
#   loop-rate-limiters           -> ID_Xsens' IK loop
#   xacro                        -> scaling the DHM xacro into a per-subject URDF
RUN pip install --no-cache-dir pin-pink qpsolvers quadprog loop-rate-limiters xacro

WORKDIR /opt/frame
COPY frame-core       /opt/frame/frame-core
COPY frame-kinematics /opt/frame/frame-kinematics
COPY frame-batch      /opt/frame/frame-batch

# --no-deps on the last two: conda already provides numpy/scipy/pandas/matplotlib, and pip wheels
# would drag a second, incompatible BLAS in beside Pinocchio's.
RUN pip install --no-cache-dir /opt/frame/frame-core \
 && pip install --no-cache-dir --no-deps /opt/frame/frame-kinematics \
 && pip install --no-cache-dir --no-deps /opt/frame/frame-batch

ENV PYTHONPATH=/opt/frame
ENV MPLBACKEND=Agg

RUN python -c "import frame_batch, frame_kinematics, frame_core, pinocchio, pink; print('image OK')"

ENTRYPOINT ["frame-batch"]
CMD ["--help"]
