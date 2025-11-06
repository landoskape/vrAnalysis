# vrAnalysis Documentation

Welcome to the documentation for **vrAnalysis**, which is the Python codebase I use for processing and analyzing virtual reality (VR) behavioral and imaging experiments. This package is not "software" that works out of the box, it's simply a collection of code I've written to analyze my data. I hope I designed it well enough that it can be used by others, which is why I've made this documentation system. So if it isn't working for you, that might be expected! There's a lot of powerful stuff in here, so hopefully it's useful, and if you really think something in here is useful to you but you can't figure it out, let me know and I'll help you get it working.

Also: if you are reading the documentation and think I could improve it in some way, please tell me. Even if I don't have time to make the changes right away, I appreciate the feedback and will try eventually.

## Overview

vrAnalysis is designed to work with:

- **Behavioral data** from [vrControl](https://github.com/landoskape/vrControl) experiments
- **Imaging data** processed with [suite2p](https://github.com/MouseLand/suite2p)
- **Database management** using Microsoft Access databases (or other SQL databases with minimal modifications)
- **Session tracking** following the [Alyx](https://github.com/cortex-lab/alyx) directory structure
- **ROI Tracking** using [ROICaT](https://github.com/landoskape/ROICaT)

## Key Features

The goal of the package (aka my coding philosophy) is to make it easy to analyze data from VR experiments. Analysis is sometimes frustrating because it's slow, so I take the time to write good code that is efficient and has clear and simple interfaces. From my perspective, it's much better for a notebook (ipynb) file to be a simple clear and obvious example of how to use the code, rather than a complex hack of a bunch of things that eventually makes plots. Becuase of that, the code is sometimes implemented in complex ways which I thought were necessary to make the interface simple and clear and fast. 

Here are some of the key features that it makes possible:

- **Database Management**: Track and manage VR session data with an easy-to-use database interface
- **Session Registration**: Automated preprocessing of behavioral and imaging data
- **Data Processing**: Generate spike maps, occupancy maps, and other spatial representations
- **Cell Tracking**: Track cells across sessions for longitudinal analysis
- **Multi-session Analysis**: Analyze data across groups of sessions

## Quick Navigation

- **[Installation Guide](installation.md)** - Get started with vrAnalysis
- **[Workflows](workflows/index.md)** - Learn how the package can be used

## Getting Help

If this documentation is not enough, there's a few other ways to get help. Firstly, 
just send me a slack message or email me. If you can't do that open a GitHub issue.
In fact, it's better to open a GitHub issue anyway! Then future people with similar
problems can benefit from finding the solution. 
