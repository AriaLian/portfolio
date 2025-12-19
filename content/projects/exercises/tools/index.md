+++
title = "Game Tools for Data Driven Design"
summary = "Tools developed to support the game production pipeline while learned to implement data-driven game frameworks."
description = ""
featuredImage = "unity1.png"
tags = ["unity", "database"]
categories = ["exercises"]
collections = ["Unity Prototype"]
draft = false
+++

## WPF Data Editor
This project presents a WPF-based game object editor designed for a space war/adventure game.

The editor allows users to create, edit and delete the following:
- Space ships
- Officers
- Planetary Systems
- Missions

The editor supports online data storage and retrieval from a MongoDB database.

There are some relationships between the objects, so related objects can automatically update when changes to one object are made.

{{< button href="https://github.com/arialian/WPF_Data_Editor" target="_self" >}}
{{< icon "github" >}} View on Github
{{< /button >}}

###
{{< gallery >}}
  <img src="wpf1.png" class="grid-w50" description="Download Database from MongoDB"/>
  <img src="wpf2.png" class="grid-w50" description="Database Downloaded"/>
  <img src="wpf3.png" class="grid-w50" description="Related Data Delete Warning"/>
  <img src="wpf4.png" class="grid-w50" description="Related Data Delete Warning"/>
{{< /gallery >}}


## Unity Data Editor
This project presents Unity editor windows designed for a space war/adventure game. 

The editor supports data storage and retrieval from the following:
- JSON file
- SQLite DB
- MongoDB

{{< button href="https://github.com/arialian/Unity_Data_Editor" target="_self" color="#7D2FA1">}}
{{< icon "github" >}} View on Github
{{< /button >}}

![](unity1.png "Editor Windows")
![](unity2.png "Data can be saved and downloaded from MongoDB or loaded from SQLite")