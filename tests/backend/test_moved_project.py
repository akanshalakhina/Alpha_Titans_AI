"""A finished video must still download after the project is moved.

WHAT HAPPENED
-------------
16 Oct. Five builds were listed in the studio. The topmost one played; the
other four showed "404 Not Found". Every one of them had its video sitting
on disk, right where it was written:

    outputs/0e932c0a1ef2/ALPHA_TITANS_horizontal.mp4     451 MB, present
    GET /api/jobs/0e932c0a1ef2/download                  404 Not Found

The thumbnails loaded. The traces loaded. Only the video was missing, and
only for the older builds.

`Job.output["path"]` holds an ABSOLUTE path, recorded when the render
finished. The project moved from

    C:\\Users\\Akansha\\Downloads\\alpha_titans\\        to      E:\\alpha_titans\\

on 14 Oct, and every build from before the move still carries a `C:` path
in its `job.json`. The file it names has not existed since the move. The
endpoint looked there, found nothing and said so - correctly, and
uselessly, because the video was two directories away under the name it
already knew.

THE RULE THIS ESTABLISHES
-------------------------
The build directory is the authority on where a build's files are; the
recorded path is a convenience. When they disagree, the directory wins.
An absolute path written into a record is a fact about a machine at a
moment, and this project has already outlived one such moment.

Nothing here invents a file. A build whose video is genuinely gone still
404s, and says that it is gone.
"""
import json
import shutil

import pytest

import adapter


@pytest.fixture
def finished(client, job):
    """A job marked done, with a real file in its build dir and a path
    recorded the way a real render records one."""
    import jobs as jobs_mod

    build = adapter.build_dir(job)
    build.mkdir(parents=True, exist_ok=True)
    video = build / "ALPHA_TITANS_horizontal.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)

    j = jobs_mod.get(job)
    j.status = "done"
    j.output = {"path": str(video), "duration": 12.0,
                "thumbnail": "thumbnail.jpg"}
    return job, build, video


def _move_away(job, build, video):
    """Rewrite the record to the pre-move location, as job.json has it."""
    import jobs as jobs_mod
    jobs_mod.get(job).output["path"] = (
        r"C:\Users\Akansha\Downloads\alpha_titans\outputs"
        "\\" + job + "\\" + video.name)


# ------------------------------------------------------- the bug itself

def test_a_video_downloads_when_the_recorded_path_is_right(client, finished):
    job, _build, _video = finished
    r = client.get(f"/api/jobs/{job}/download")
    assert r.status_code == 200
    assert r.content[4:8] == b"ftyp"


def test_a_video_still_downloads_after_the_project_moved(client, finished):
    """The four builds that showed 404 in the studio."""
    job, build, video = finished
    _move_away(job, build, video)

    r = client.get(f"/api/jobs/{job}/download")

    assert r.status_code == 200, (
        "the file is in the build directory under the name the record "
        "already knows; a dead C: path must not hide it")
    assert r.content[4:8] == b"ftyp"


def test_the_moved_file_is_served_from_this_projects_outputs(client,
                                                              finished):
    """Not from anywhere the stale record points - from here."""
    job, build, video = finished
    _move_away(job, build, video)
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"THIS-ONE")

    r = client.get(f"/api/jobs/{job}/download")
    assert r.content.endswith(b"THIS-ONE")


# ---------------------------------------- and what must NOT start working

def test_a_genuinely_missing_video_still_says_so(client, finished):
    """Falling back must not become guessing. A build whose video was
    deleted has no video, and the person needs to be told that."""
    job, build, video = finished
    _move_away(job, build, video)
    video.unlink()

    r = client.get(f"/api/jobs/{job}/download")
    assert r.status_code == 404


def test_a_build_with_no_output_recorded_is_unchanged(client, job):
    """Still rendering, or never started. 409, as before."""
    r = client.get(f"/api/jobs/{job}/download")
    assert r.status_code == 409


def test_the_fallback_cannot_reach_outside_the_build_directory(client,
                                                                finished):
    """A recorded name is still a name from a file we wrote, but the
    rule has to hold whatever the record says."""
    job, build, video = finished
    import jobs as jobs_mod
    jobs_mod.get(job).output["path"] = (
        r"C:\somewhere\else\..\..\Windows\System32\config\SAM")

    r = client.get(f"/api/jobs/{job}/download")
    assert r.status_code == 404
    assert "SAM" not in r.text


def test_a_differently_named_video_is_found_too(client, finished):
    """Titles change the filename: `Motorised_LEGO_build_horizontal.mp4`
    is as real a name as the default one."""
    job, build, video = finished
    renamed = build / "Motorised_LEGO_build_horizontal.mp4"
    shutil.move(str(video), str(renamed))
    import jobs as jobs_mod
    jobs_mod.get(job).output["path"] = (
        r"C:\Users\Akansha\Downloads\alpha_titans\outputs"
        "\\" + job + "\\" + renamed.name)

    r = client.get(f"/api/jobs/{job}/download")
    assert r.status_code == 200 and r.content[4:8] == b"ftyp"


def test_the_listing_and_the_download_agree(client, finished):
    """A build the studio shows as playable must be playable. These
    disagreed for four builds, which is how the bug was noticed."""
    job, build, video = finished
    _move_away(job, build, video)

    listed = client.get("/api/jobs").json()["jobs"]
    row = next((j for j in listed if j["id"] == job), None)
    assert row is not None and row["status"] == "done"
    assert client.get(f"/api/jobs/{job}/download").status_code == 200
