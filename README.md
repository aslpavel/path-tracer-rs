This is simple path tracer with following properties
- [X] Only lambertian diffusion for now
- [X] Signed distance function defined objects
- [X] Ray marching tracing
- [X] No performance optimazation whatsoever

## Running
- insall `carog-script`
```
cargo install cargo-script
```
- run
```
# output will be stored in `path-tracer-output.png`
./path-tracer.rs
```

## Output
Samples: 1024
Bounces: 3

![s1024b3](/images/s1024b3.png "samples 1024, bounces 3")

Samples: 4096
Bounces: 3

![s4096b3](/images/s4096b3.png "samples 4096, bounces 3")


Samples: 32768
Bounces: 4

![s32768b4](/images/s32768b4.png "samples 32768, bounces 4")
